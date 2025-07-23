"""
数据参考：https://github.com/facebookresearch/simsiam/blob/main/simsiam/loader.py
模型参考：https://github.com/facebookresearch/simsiam/blob/main/simsiam/builder.py，ultralytics.nn.tasks.Basemodel
训练参考：ultralytics.engine.Basetrainer
"""
import argparse
import os
import math
import random
import contextlib
import time
import io
import pandas as pd
from copy import deepcopy
from pathlib import Path
from PIL import ImageFilter, Image
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from scipy.ndimage import gaussian_filter1d

from ultralytics.nn.tasks import yaml_model_load
from ultralytics.cfg import get_save_dir
from ultralytics.utils import plt_settings, LOGGER, colorstr
from ultralytics.utils.torch_utils import model_info
from ultralytics.utils.ops import make_divisible
from ultralytics.nn.modules import Conv, C3k2


def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    import ast

    # Args
    legacy = True  # backward compatibility for v3/v5/v8/v9 models
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = torch.nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d["backbone"]):  # from, number, module, args
        m = (
            getattr(torch.nn, m[3:])
            if "nn." in m
            else getattr(__import__("torchvision").ops, m[16:])
            if "torchvision.ops." in m
            else globals()[m]
        )  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in {Conv, C3k2}:
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
            if m in {C3k2}:
                args.insert(2, n)  # number of repeats
                n = 1
            if m is C3k2:  # for M/L/X sizes
                legacy = False
                if scale in "mlx":
                    args[3] = True
        else:
            c2 = ch[f]

        m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m_.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return torch.nn.Sequential(*layers), sorted(save), ch


class TwoCropsTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur:
    def __init__(self, sigma=None):
        if sigma is None:
            sigma = [.1, 2.]
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class ContrastiveDataset(Dataset):
    def __init__(self, root):
        super().__init__()
        self.img_list = [os.path.join(root, i) for i in os.listdir(root)]
        augmentation = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.4, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                                   p=0.8),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        self.transform = TwoCropsTransform(augmentation)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img = Image.open(img_path).convert("L")
        q, k = self.transform(img)
        return q, k, img_path


class YOLOSimSiam(nn.Module):
    def __init__(self, cfg, ch=1, verbose=True, dim=2048, pred_dim=512):
        super().__init__()
        self.yaml = yaml_model_load(cfg)
        # Define model
        self.model, self.save, self.ch = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.model.append(nn.AdaptiveAvgPool2d((1, 1)))
        # 3层的projector
        prev_dim = self.ch[-1]
        self.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                nn.BatchNorm1d(prev_dim),
                                nn.SiLU(inplace=True),  # first
                                nn.Linear(prev_dim, prev_dim, bias=False),
                                nn.BatchNorm1d(prev_dim),
                                nn.SiLU(inplace=True),  # second
                                nn.Linear(prev_dim, dim, bias=False),
                                nn.BatchNorm1d(dim, affine=False))
        # 2层的predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                       nn.BatchNorm1d(pred_dim),
                                       nn.SiLU(inplace=True),
                                       nn.Linear(pred_dim, dim))
        model_info(self, verbose=verbose, imgsz=224)

    def forward(self, x1, x2):
        z1 = self.fc(self.model(x1).flatten(1))
        z2 = self.fc(self.model(x2).flatten(1))

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return p1, p2, z1.detach(), z2.detach()


def main(args):
    """single-gpu training"""
    assert torch.cuda.is_available(), "SSL should be implement on GPU"
    device = torch.device("cuda:0")
    model = YOLOSimSiam(cfg=args.model).to(device)
    criterion = nn.CosineSimilarity(dim=1).to(device)
    train_dataset = ContrastiveDataset(root=args.data)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    knn_train_loader = DataLoader(dataset=datasets.ImageFolder(args.knn_train,
                                                               transform=transforms.Compose([
                                                                   transforms.Grayscale(),
                                                                   transforms.Resize(256),
                                                                   transforms.CenterCrop(224),
                                                                   transforms.ToTensor(),
                                                               ])),
                                  batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    knn_val_loader = DataLoader(dataset=datasets.ImageFolder(args.knn_val,
                                                             transform=transforms.Compose([
                                                                 transforms.Grayscale(),
                                                                 transforms.Resize(256),
                                                                 transforms.CenterCrop(224),
                                                                 transforms.ToTensor(),
                                                             ])),
                                batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    nb = len(train_loader)
    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params=optim_params, lr=args.lr0)
    lf = lambda x: (1 + math.cos(x * math.pi / args.epochs)) * 0.5 * (1 - args.lrf) + args.lrf
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lf)

    start_time = time.time()
    best_fit, lr, tloss, l_std, acc = 0., 0., 0., 0., 0.
    save_dir = get_save_dir(args)
    weight_dir = save_dir / "weights"
    weight_dir.mkdir(parents=True, exist_ok=True)
    # last, best = weight_dir / "last.pt", weight_dir / "best.pt"
    result_file = save_dir / "results.csv"

    print(f"Using {train_loader.num_workers} dataloader workers")
    print(f"Logging results to {colorstr('bold', save_dir)}")
    print(f"start training for {args.epochs} epochs...")
    for j in range(args.epochs):
        epoch_start_time = time.time()
        model.train()
        print("\n" + "%11s" * 5 % ("Epoch", "GPU_mem", "Loss", "L_std", "lr"))
        pbar = tqdm(train_loader, total=nb)
        # warm up in 1st epoch
        warmup_scheduler = None
        if j == 0:
            wf = 1.0 / 1000
            wi = min(1000, len(train_loader) - 1)
            warmup_lf = lambda x: 1 if x >= wi else wf * (1 - x / wi) + x / wi
            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warmup_lf)
        for i, (qs, ks, _) in enumerate(pbar):
            qs = qs.to(device)
            ks = ks.to(device)
            p1, p2, z1, z2 = model(x1=qs, x2=ks)
            loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
            tloss = (tloss * i + loss.item()) / (i + 1) if tloss is not None else loss.item()
            # 参考SimSiam计算某个样本的channel std
            z_0 = z1[0]
            z_l2 = torch.norm(z_0)
            z_norm = z_0 / z_l2
            z_std = torch.std(z_norm)
            l_std = z_std.item()
            lr = optimizer.param_groups[0]["lr"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if warmup_scheduler:
                warmup_scheduler.step()

            pbar.set_description(
                ("%11s" * 2 + "%11.4g" * 3)
                % (f"{j + 1}/{args.epochs}", f"{torch.cuda.memory_reserved() / (2 ** 30):.3g}G", tloss, l_std, lr)
            )
        scheduler.step()

        # 进行KNN评估
        acc = knn_classifier(train_loader=knn_train_loader, val_loader=knn_val_loader, model=model.model, k=5)
        print(f"Epoch [{j + 1}] k-NN Acc: {acc:.2%}")
        # log
        epoch_time = time.time() - epoch_start_time
        if best_fit > tloss:
            best_fit = tloss
        log_list = [round(x, 5) for x in [j, lr, tloss, l_std, acc, epoch_time]]
        with open(result_file, "a") as f:
            s = (("%s," * 6 % ("Epoch", "lr", "Loss", "L_std", "KNN_acc", "Time")).rstrip(",") + "\n") if j == 0 else ""
            f.write(s + ("%.6g," * 6 % tuple(log_list)).rstrip(",") + "\n")
        # save model
        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": j,
                "fitness": tloss,
                "model": model
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()
        # last.write_bytes(serialized_ckpt)
        # if best_fit == tloss:
        #     best.write_bytes(serialized_ckpt)
        if args.save_period > 0 and (j + 1) % args.save_period == 0:
            (weight_dir / f"epoch{j + 1}.pt").write_bytes(serialized_ckpt)

    print(f"total training time {(time.time() - start_time) / 3600:.3f} hours")
    plot_result(result_file)


@plt_settings()
def plot_result(file):
    save_dir = Path(file).parent
    fig, ax = plt.subplots(2, 2, figsize=(12, 6), tight_layout=True)
    index = [1, 2, 3, 4]
    ax = ax.ravel()
    try:
        data = pd.read_csv(file)
        s = [x.strip() for x in data.columns]
        x = data.values[:, 0]
        for i, j in enumerate(index):
            y = data.values[:, j].astype("float")
            ax[i].plot(x, y, marker=".", label=file.stem, linewidth=2, markersize=8)  # actual results
            ax[i].plot(x, gaussian_filter1d(y, sigma=3), ":", label="smooth", linewidth=2)  # smoothing line
            ax[i].set_title(s[j], fontsize=12)
    except Exception as e:
        LOGGER.error(f"Plotting error for {file}: {e}")
    ax[1].legend()
    fname = save_dir / "results.png"
    fig.savefig(fname, dpi=200)
    plt.close()


def knn_classifier(train_loader, val_loader, model, k=1):
    """
    Args:
        train_loader: 训练集数据加载器 (包含标签)
        val_loader: 验证集数据加载器 (包含标签)
        model: 预训练模型 (backbone部分)
        k: 近邻数量
    Returns:
        准确率
    """
    model.eval()
    device = next(model.parameters()).device

    # 1. 构建训练特征库
    train_features = []
    train_labels = []
    with torch.no_grad():
        for images, labels in tqdm(train_loader, desc="Building train feature bank"):
            images = images.to(device)
            features = model(images).flatten(1)  # 获取骨干网络输出
            features = nn.functional.normalize(features, dim=1)
            train_features.append(features.cpu())
            train_labels.append(labels)

    train_features = torch.cat(train_features, dim=0).numpy()
    train_labels = torch.cat(train_labels, dim=0).numpy()

    # 2. 验证集评估
    val_features = []
    val_labels = []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Building val feature bank"):
            images = images.to(device)
            features = model(images).flatten(1)
            features = nn.functional.normalize(features, dim=1)
            val_features.append(features.cpu())
            val_labels.append(labels)

    val_features = torch.cat(val_features, dim=0).numpy()
    val_labels = torch.cat(val_labels, dim=0).numpy()

    # 3. 构建knn分类器进行评估
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", n_jobs=-1, weights="distance")
    knn.fit(train_features, train_labels)

    pred_labels = knn.predict(val_features)
    acc = accuracy_score(val_labels, pred_labels)
    return acc


parser = argparse.ArgumentParser("contrastive training only for GPUs")
# data
parser.add_argument("--data", default="/home/featurize/Leakage_images")
parser.add_argument("--knn_train", default="/home/featurize/Leakage_KNN/train")
parser.add_argument("--knn_val", default="/home/featurize/Leakage_KNN/val")
# model
parser.add_argument("--model", default="/home/featurize/work/YOLO_model/cfg/yolo11s.yaml")
# train
parser.add_argument("--epochs", default=200)
parser.add_argument("--workers", default=8)
parser.add_argument("--batch-size", default=128)
parser.add_argument("--save-period", default=50)
# optimizer
parser.add_argument("--lr0", default=0.01, type=float, help="initial learning rate")
parser.add_argument("--lrf", default=0.01, type=float, help="lr decay factor")
# log
parser.add_argument("--exist-ok", default=False, type=bool,
                    help="whether to overwrite existing experiment")
parser.add_argument("--project", default="knn_contra", type=str, help="name of save dir")
parser.add_argument("--name", default="0.01_adam_cos", type=str,
                    help="subdirectory within project folder, logs and outputs are stored.")


if __name__ == '__main__':
    ps = parser.parse_args()
    main(ps)
