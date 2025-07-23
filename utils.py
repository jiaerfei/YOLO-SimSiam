"""
参考：MAE、YOLOv8、Labelme等github库的utils
以及一些自定义的
"""
import json
import math
import cv2
import numpy as np
import shutil
import os
import piexif
import glob
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from lxml import etree
from pathlib import Path
from copy import deepcopy
import PIL
import PIL.ExifTags
import PIL.ImageOps
import PIL.Image
from PIL import Image


from ultralytics.utils import YAML


def cut_image(src, h_num, w_num, dst):
    """
    src: 图片文件夹路径
    img_name: 图片名称
    h_num: 高度方向裁剪份数
    w_num: 宽度方向裁剪份数
    dst: 保存文件夹路径
    """
    pbar = tqdm(os.listdir(src))
    for img in pbar:
        pbar.set_description(f"processing {img}")
        img_array = cv2.imread(os.path.join(src, img))
        img_h, img_w = img_array.shape[0], img_array.shape[1]
        cut_h, cut_w = int(img_h / h_num), int(img_w / w_num)

        h_d = 0
        for h_i in range(h_num):
            w_d = 0
            for w_j in range(w_num):
                img_cut = img_array[h_d:h_d + cut_h, w_d:w_d + cut_w]
                # 按照裁剪图像在原图位置进行命名，行列(h, w)
                cv2.imwrite(os.path.join(dst, img.split(".")[0] + "_{}_{}.jpg".format(h_i, w_j)), img_cut)
                w_d += cut_w
            h_d += cut_h


def apply_exif_orientation(image_path, remove_exif=False):
    """
    image_path: 图像路径
    remove_exif: 是否移除exif信息
    """
    image = PIL.Image.open(image_path)
    exif = image.getexif()
    exif = {PIL.ExifTags.TAGS[k]: v for k, v in exif.items() if k in PIL.ExifTags.TAGS}
    if remove_exif:
        piexif.remove(image)
        print(image_path)

    orientation = exif.get("Orientation", None)
    if orientation == 1:
        # do nothing
        return image
    elif orientation == 2:
        # left-to-right mirror
        return PIL.ImageOps.mirror(image)
    elif orientation == 3:
        # rotate 180
        return image.transpose(PIL.Image.ROTATE_180)
    elif orientation == 4:
        # top-to-bottom mirror
        return PIL.ImageOps.flip(image)
    elif orientation == 5:
        # top-to-left mirror
        return PIL.ImageOps.mirror(image.transpose(PIL.Image.ROTATE_270))
    elif orientation == 6:
        # rotate 270
        return image.transpose(PIL.Image.ROTATE_270)
    elif orientation == 7:
        # top-to-right mirror
        return PIL.ImageOps.mirror(image.transpose(PIL.Image.ROTATE_90))
    elif orientation == 8:
        # rotate 90
        return image.transpose(PIL.Image.ROTATE_90)
    else:
        return image


def copy_exif_image(file_path, save_path):
    """
    遍历文件夹下所有图像，根据其exif信息进行处理，并将其统一复制到固定文件夹
    """
    i = 0
    for root, dirs, files in os.walk(file_path):
        j = 0
        for file in tqdm(files):
            img_path = os.path.join(root, file)
            try:
                image_exif = apply_exif_orientation(img_path)
                new_name = "STDD_{}_{}_{}".format(i, j, file)
                new_path = os.path.join(save_path, new_name)
                image_exif.save(new_path, dpi=(300, 300))
                j += 1
            except Exception as e:
                print(e)
                continue
        i += 1


def read_excel_plot(excel_path, save_path, name_path=None, data_num=6):
    """
    读取地层cpt数据进行图像绘制，标注分层位置与对应地层类别
    excel_path: 读取的excel文件路径
    save_path: 图片保存的文件夹路径
    name_path：土体类别名称对应编号表
    data_num:  excel表内钻孔数据的间隔
    """
    data = pd.read_excel(excel_path, sheet_name=None, header=2)
    class_name = pd.read_excel(name_path, keep_default_na=False)
    hc = class_name.shape[0]
    name_num_dict = {}
    for c in range(hc):
        num, name = class_name.iloc[c]
        if num:
            name_num_dict[name] = num
    # print(name_num_dict)
    sheet_num = len(data)
    sheet_list = list(data.keys())
    for i in range(sheet_num):
        d = data[sheet_list[i]]
        h, w = d.shape
        total_num = int((w + 1) / data_num)
        for j in tqdm(range(total_num)):
            try:
                # 读取数据
                d_j = d.iloc[:, j * data_num:j * data_num + data_num]
                # 绘制cpt曲线
                fig, ax = plt.subplots(figsize=(8, 16))
                ax.xaxis.set_ticks_position("top")
                ax.invert_yaxis()
                ax.set_xlim(-1, 30)
                if j > 0:
                    d_j["x"] = d_j[f"x.{j}"]
                    d_j["y"] = d_j[f"y.{j}"]
                ax.plot([x for x in d_j["x"].dropna()],
                        [y for y in d_j["y"].dropna()], linewidth=3.0)
                ax.axis("off")
                # 进行标记
                if j > 0:
                    d_j["层底"] = d_j[f"层底.{j}"]
                    d_j["土层名称"] = d_j[f"土层名称.{j}"]
                for m in zip([e for e in list(d_j["层底"].dropna())],
                             [name_num_dict[t] for t in list(d_j["土层名称"].dropna())]):
                    h, n = m
                    ax.text(-1, h, n)
                    ax.scatter(-1, h)
                # 保存图片
                root_name = excel_path.split("\\")[-1].split(".")[0]  # 根据需求自行修改
                img_name = f"{root_name}_{sheet_list[i]}_{j}.jpg"
                plt.savefig(os.path.join(save_path, img_name), bbox_inches="tight", pad_inches=0, dpi=300)
                plt.close()  # 重复绘制，避免重复建立fig对象
            except Exception as e:
                print(f"error{sheet_list[i]}_{j}", e)
                pass


def parse_xml_to_dict(xml_doc):
    """
    xml_doc: 利用read打开后读取得到str文件，再由etree转换为xml格式传入
    return: 字典形式的xml内容
    """
    if len(xml_doc) == 0:
        return {xml_doc.tag: xml_doc.text}

    result = {}
    for child in xml_doc:
        child_res = parse_xml_to_dict(child)
        if child.tag != "object":
            result[child.tag] = child_res[child.tag]
        else:  # 可能存在多个object，也就是标注框
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_res[child.tag])
    return {xml_doc.tag: result}


def xml2yolo(data_cfg, name="train"):
    data = YAML.load(data_cfg)
    class_dict = data["names"]
    name_class = {}
    for k, v in deepcopy(class_dict).items():
        name_class[str(v)] = k

    path = Path(data["path"]) / name
    label_path = path / "labels_xml"
    for n in tqdm(os.listdir(label_path), desc=f"translating {name} info"):
        with open(str(label_path / n), "r", errors="ignore") as file:
            xml_str = file.read()
            file.close()
        xml_doc = etree.fromstring(xml_str)
        xml_dict = parse_xml_to_dict(xml_doc)
        img_w = int(xml_dict["annotation"]["size"]["width"])
        img_h = int(xml_dict["annotation"]["size"]["height"])

        label_file = n.split(".")[0] + ".txt"
        label_file = path / "labels" / label_file
        with open(str(label_file), "w") as f:
            for obj in xml_dict["annotation"]["object"]:
                xmin = float(obj["bndbox"]["xmin"])
                xmax = float(obj["bndbox"]["xmax"])
                ymin = float(obj["bndbox"]["ymin"])
                ymax = float(obj["bndbox"]["ymax"])
                class_idx = name_class[obj["name"]]

                xc = xmin + (xmax - xmin) / 2
                yc = ymin + (ymax - ymin) / 2
                w = xmax - xmin
                h = ymax - ymin

                xc = round(xc / img_w, 6)
                yc = round(yc / img_h, 6)
                w = round(w / img_w, 6)
                h = round(h / img_h, 6)

                info = [str(i) for i in [class_idx, xc, yc, w, h]]
                f.write(" ".join(info) + "\n")


def json2yolo(data_cfg, name="train", detect=False, segment=False):
    """
    将json格式标签转为yolo格式，数据集文件夹组成形式如下
    -root
      -train
        -images
        -labels
      -val
        -images
        -labels
    """
    assert detect or segment, "should clarify detect or segment"

    data = YAML.load(data_cfg)
    class_dict = data["names"]
    name_class = {}
    for k, v in deepcopy(class_dict).items():
        name_class[str(v)] = k

    label_path = Path(data["path"]) / name / "images"
    # label_path = Path(data["path"]) / name / "with_json"
    json_list = glob.glob(os.path.join(label_path, "*.json"))
    for n in tqdm(json_list, desc=f"translating {name} info"):
        with open(n, "r", errors="ignore") as file:
            json_dict = json.load(file)
            file.close()
        img_w = json_dict["imageWidth"]
        img_h = json_dict["imageHeight"]

        label_file = os.path.join(label_path, Path(n).stem + ".txt")
        label_file = label_file.replace("images", "labels")
        # label_file = label_file.replace("with_json", "labels")
        with open(str(label_file), "w") as f:
            for obj in json_dict["shapes"]:
                try:
                    class_idx = name_class[obj["label"]]
                except KeyError:
                    print(n)

                if segment:
                    points = np.asarray(obj["points"]).flatten().tolist()
                    norm_points = [round(x / (img_w if i % 2 == 0 else img_h), 6) for i, x in enumerate(points)]
                    info = [str(i) for i in [class_idx] + norm_points]

                if detect:
                    (xmin, ymin), (xmax, ymax) = obj["points"]
                    assert xmax > xmin and ymax > ymin, f"Warning: in '{n}' json, there are some bbox w/h <=0"
                    xc = xmin + (xmax - xmin) / 2
                    yc = ymin + (ymax - ymin) / 2
                    w = xmax - xmin
                    h = ymax - ymin

                    xc = round(xc / img_w, 6)
                    yc = round(yc / img_h, 6)
                    w = round(w / img_w, 6)
                    h = round(h / img_h, 6)

                    info = [str(i) for i in [class_idx, xc, yc, w, h]]

                f.write(" ".join(info) + "\n")


def compute_mean_std(image_folder):
    # 初始化变量
    sum_r, sum_g, sum_b = 0.0, 0.0, 0.0
    sum_sq_r, sum_sq_g, sum_sq_b = 0.0, 0.0, 0.0
    total_pixels = 0

    # 遍历所有图像文件
    for root, dirs, files in os.walk(image_folder):
        for file in tqdm(files):
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                path = os.path.join(root, file)
                try:
                    # 读取图像并转换为RGB
                    img = Image.open(path).convert('RGB')
                    img_array = np.array(img, dtype=np.float64) / 255.0
                    h, w, _ = img_array.shape
                    pixels = h * w
                    total_pixels += pixels

                    # 累加各通道的和与平方和
                    sum_r += np.sum(img_array[:, :, 0])
                    sum_g += np.sum(img_array[:, :, 1])
                    sum_b += np.sum(img_array[:, :, 2])

                    sum_sq_r += np.sum(img_array[:, :, 0] ** 2)
                    sum_sq_g += np.sum(img_array[:, :, 1] ** 2)
                    sum_sq_b += np.sum(img_array[:, :, 2] ** 2)
                except Exception as e:
                    print(f"Error processing {path}: {e}")

    # 计算均值
    mean_r = sum_r / total_pixels
    mean_g = sum_g / total_pixels
    mean_b = sum_b / total_pixels

    # 计算标准差
    std_r = np.sqrt((sum_sq_r / total_pixels) - (mean_r ** 2))
    std_g = np.sqrt((sum_sq_g / total_pixels) - (mean_g ** 2))
    std_b = np.sqrt((sum_sq_b / total_pixels) - (mean_b ** 2))

    print(f"均值 (R, G, B): {(mean_r, mean_g, mean_b)}")
    print(f"标准差 (R, G, B): {(std_r, std_g, std_b)}")


def resize_to_show(image, name):
    """将图像缩放到适合屏幕的尺寸"""
    h, w = image.shape[:2]
    cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
    cv2.resizeWindow(name, w, h)  # (width, height)
    cv2.imshow(name, image)




