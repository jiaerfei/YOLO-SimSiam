
import cv2
from collections import OrderedDict
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.models._utils import IntermediateLayerGetter


from ultralytics.utils.plotting import plt_settings
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.models.yolo.segment import SegmentationTrainer

from SSL_main import YOLOSimSiam, ContrastiveDataset

from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


@plt_settings()
def plot_contra_images(root, save_path="runs/contra", bs=4):
    """display one batch images and their 2 views"""
    dataset = ContrastiveDataset(root)
    loader = DataLoader(dataset, bs, collate_fn=dataset.collate_fn)
    fig, axs = plt.subplots(bs, 3, figsize=(8, 6), tight_layout=True)
    for batch in loader:
        for i in range(bs):
            im0 = Image.open(batch[2][i])
            im1 = batch[0][i].permute(1, 2, 0) * 255
            im2 = batch[1][i].permute(1, 2, 0) * 255

            axs[i, 0].axis("off")
            axs[i, 0].imshow(im0)
            # axs[i, 0].set_title("original image")

            axs[i, 1].axis("off")
            axs[i, 1].imshow(im1)
            # axs[i, 1].set_title("view 1 mage")

            axs[i, 2].axis("off")
            axs[i, 2].imshow(im2)
            # axs[i, 2].set_title("view 2 mage")
        break  # one batch is over
    plt.savefig(Path(save_path) / "pair_images.png", dpi=300, bbox_inches="tight")
    # plt.show()
    plt.close()


### 直接可视化中间特征图，直接输出
class ILGetConcat(IntermediateLayerGetter):
    """yolo检测系列模型涉及concat操作，需要继承重写"""
    def forward(self, x, save=None):
        y = []
        out = OrderedDict()
        for name, module in self.items():
            if module.f != -1:
                x = [x if j == -1 else y[j] for j in module.f]  # from earlier layers
            x = module(x)
            y.append(x if module.i in save else None)  # save output
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


# 参考：https://blog.csdn.net/m0_46412065/article/details/127882492
def visualize_feature_map(checkpoint, image_path, n=4, layer="6", task="contra", save=False):
    """针对YOLO模型可视化其中间特征图"""
    # 加载带参数的模型
    yolo_model, _ = attempt_load_one_weight(checkpoint)
    model = yolo_model.model.cuda()  # 到这一层才能显示children
    # 输出中间特征图
    new_model = ILGetConcat(model, {layer: "feat1"})
    transform = transforms.Compose([transforms.Resize((640, 640)), transforms.ToTensor()])
    img = Image.open(image_path).convert("L")
    img_tensor = transform(img).unsqueeze(0).cuda()
    out = new_model(img_tensor, yolo_model.save)

    # 可视化
    out = out["feat1"].detach().cpu().squeeze(0).permute(1, 2, 0)
    # 获取全局最小最大值用于统一颜色映射
    vmin = out.min()
    vmax = out.max()
    fig, axs = plt.subplots(n, n)
    axs = axs.ravel()  # 从二维（4，8）矩阵形式拉直变为一维32，便于索引
    for i in range(n*n):
        img = axs[i].imshow(out[..., i*n], cmap='viridis', vmin=vmin, vmax=vmax)
        axs[i].axis("off")

    fig.colorbar(img, ax=axs)

    if save:
        plt.savefig(f"{Path(image_path).stem}_{task}_{layer}_feature_map.png", dpi=200, bbox_inches="tight")
    fig.suptitle(f"{Path(image_path).stem}_{layer}")
    plt.show()


# 参考grad_cam绘制对比学习的热力图
# https://jacobgil.github.io/pytorch-gradcam-book/Pixel%20Attribution%20for%20embeddings.html
# 对比两张图像特征相似的地方
# 这个例子是处理单通道的图像，如处理彩色图像需进行简单修改
class YOLOSimSiamFeatureExtractor(nn.Module):
    """一种处理方式，让输出满足GradCam的计算过程"""
    def __init__(self, model):
        super().__init__()
        self.model = model.model

    def __call__(self, x):
        return self.model(x)[:, :, 0, 0]


class SimilarityToConceptTarget:
    def __init__(self, features):
        self.features = features

    def __call__(self, model_output):
        cos = nn.CosineSimilarity(dim=0)
        return cos(model_output, self.features)


def get_image(image_path, size=(512, 512), mode="L"):
    """此处示例为单通道图像，读取模式为L，且mean和std长度设置为1"""
    img = np.array(Image.open(image_path).convert(mode))
    img = cv2.resize(img, size)
    img_float = np.float32(img) / 255
    preprocessing = transforms.ToTensor()
    input_tensor = preprocessing(img.copy()).unsqueeze(0)
    return img, img_float, input_tensor


def visualize_contra_map(checkpoint, image_path, target_path, save=False):
    assert torch.cuda.is_available(), "SSL should be implement on GPU"
    yolo_model, _ = attempt_load_one_weight(checkpoint)
    model = YOLOSimSiamFeatureExtractor(yolo_model).cuda()
    # print(model)

    image, image_float, image_tensor = get_image(image_path)
    target, target_float, target_tensor = get_image(target_path)

    # target_layers = [model.model[-2]]  # last layer before avgpool with gradients
    target_layers = [model.model[-4]]  # 取更浅层的
    target_features = model(target_tensor.cuda())[0, :]  # should be scalar for gradients
    contra_targets = [SimilarityToConceptTarget(target_features)]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        target_cam = cam(input_tensor=image_tensor, targets=contra_targets)[0, :]
    target_cam_image = show_cam_on_image(image_float[:, :, None], target_cam, use_rgb=True)
    plt.imshow(target_cam_image)
    plt.title(f"{Path(image_path).stem} vs {Path(target_path).stem}")
    if save:
        plt.savefig(f"{Path(image_path).stem} vs {Path(target_path).stem}.png", dpi=200, bbox_inches="tight")
    plt.show()


# 参考grad_cam绘制分割任务的热力图
# 针对YOLO分割模型的
# 这个例子是处理彩色图像，如处理单通道图像需进行简单修改
class YOLOSegExtractor(nn.Module):
    """一种处理方式，让输出满足GradCam的计算过程"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def __call__(self, x):
        return self.model(x)[0]


def visualize_seg_map(checkpoint, image_path, save=False):
    assert torch.cuda.is_available(), "SSL should be implement on GPU"
    yolo_model, _ = attempt_load_one_weight(checkpoint)
    model = YOLOSegExtractor(yolo_model).cuda()

    image, image_float, image_tensor = get_image(image_path, (640, 640), mode="RGB")

    target_layers = [yolo_model.model[-2]]  # seg层前面那一层


    # EigenCAM这个方法没有进行反向传播，因为检测、实例分割模型的损失函数比较复杂
    # 如果要纳入梯度，需要和对比学习一样，自己构造target，来计算损失函数并进行反传
    with EigenCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=image_tensor)[0, :]
    cam_image = show_cam_on_image(image_float, grayscale_cam, use_rgb=True)
    plt.imshow(cam_image)
    plt.title(f"{Path(image_path).stem}")
    if save:
        plt.savefig(f"{Path(image_path).stem}.png", dpi=200, bbox_inches="tight")
    plt.show()
