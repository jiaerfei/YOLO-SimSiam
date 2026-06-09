"""
参考：MAE、YOLO(ultralytics)、Labelme等github库的utils
以及一些自定义的
"""
import json
import time

import cv2
import numpy as np
import os
import glob
import pandas as pd
import torch
import matplotlib.pyplot as plt
import math

from tqdm import tqdm
from lxml import etree
from pathlib import Path
from copy import deepcopy
from numpy.typing import NDArray
from PIL import Image, ImageDraw
from ultralytics.utils import YAML
from collections import defaultdict


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
    """
    参考：https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/blob/master/pytorch_object_detection/yolov3_spp/trans_voc2yolo.py
    将xml格式标签转为yolo格式，仅针对检测任务
    """
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


def json2yolo(data_cfg, name="train", detect=False, segment=False, semantic=False):
    """
    参考：https://github.com/wkentaro/labelme/blob/main/examples/instance_segmentation/labelme2coco.py
    将json格式标签转为yolo格式，适用于检测和分割任务，数据集文件夹组成形式如下
    补充了转换为语义分割格式的代码，生成标签图像.png
    具体可参照ultralytics中data的cfg中文件设置
    -root
      -train
        -images
        -labels
        -sem_labels
      -val
        -images
        -labels
        -sem_labels
    """
    assert detect or segment or semantic, "should clarify the task"

    data = YAML.load(data_cfg)
    class_dict = data["names"]
    name_class = {}
    for k, v in deepcopy(class_dict).items():
        if detect or segment:
            name_class[str(v)] = k
        elif semantic:
            name_class[str(v)] = k + 1  # 对于语义分割，背景类别为0

    label_path = Path(data["path"]) / name / "images"
    json_list = glob.glob(os.path.join(label_path, "*.json"))
    for n in tqdm(json_list, desc=f"translating {name} info"):
        with open(n, "r", errors="ignore") as file:
            json_dict = json.load(file)
            file.close()
        img_w = json_dict["imageWidth"]
        img_h = json_dict["imageHeight"]

        if detect or segment:
            label_file = os.path.join(label_path, Path(n).stem + ".txt")
            label_file = label_file.replace("images", "labels")
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

        elif semantic:
            sem_label = os.path.join(label_path, Path(n).stem + ".png")
            sem_label = sem_label.replace("images", "sem_labels")
            img_shape = (img_h, img_w)
            cls = np.zeros(img_shape, dtype=np.int32)
            for shape in json_dict["shapes"]:
                points = shape["points"]
                label = shape["label"]
                shape_type = shape.get("shape_type", None)
                cls_id = name_class[label]
                mask = shape_to_mask(img_shape, points, shape_type)
                cls[mask] = cls_id
            cls = cls.astype(np.uint8)
            # 可选：参考PASCAL VOC，将目标边缘的像素值设置为255，避免边缘处的误导
            # border = np.zeros_like(cls, dtype=np.uint8)
            # classes = np.unique(cls[cls > 0])
            # for c in classes:
            #     c_mask = (cls == c).astype(np.uint8)
            #     contours, _ = cv2.findContours(c_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            #     cv2.drawContours(border, contours, -1, 250, 1)
            # cls[border == 250] = 255
            # 保存标签
            cv2.imwrite(sem_label, cls)


def shape_to_mask(
    img_shape: tuple[int, ...],
    points: list[list[float]],
    shape_type: str | None = None,
    line_width: int = 10,
    point_size: int = 5,
) -> NDArray[np.bool_]:
    """来源：labelme.utils.shape_to_mask"""
    mask = Image.fromarray(np.zeros(img_shape[:2], dtype=np.uint8))
    draw = ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == "circle":
        assert len(xy) == 2, "Shape of shape_type=circle must have 2 points"
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == "rectangle":
        assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"
        (x0, y0), (x1, y1) = xy
        draw.rectangle(
            ((min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1))),
            outline=1,
            fill=1,
        )
    elif shape_type == "line":
        assert len(xy) == 2, "Shape of shape_type=line must have 2 points"
        draw.line(xy=xy, fill=1, width=line_width)  # type: ignore[arg-type]
    elif shape_type == "linestrip":
        draw.line(xy=xy, fill=1, width=line_width)  # type: ignore[arg-type]
    elif shape_type == "point":
        assert len(xy) == 1, "Shape of shape_type=point must have 1 points"
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    elif shape_type in [None, "polygon"]:
        assert len(xy) > 2, "Polygon must have points more than 2"
        draw.polygon(xy=xy, outline=1, fill=1)  # type: ignore[arg-type]
    else:
        raise ValueError(f"shape_type={shape_type!r} is not supported.")
    return np.array(mask, dtype=bool)


def count_pixel_num(path=r"C:\jiafeidl\YOLO_model_local\area"):
    # 得到分割结果图后，用这个获得非0像素数量，也就是前景
    # 这里假设图像中只有一个目标，如果有多个，需要一些额外操作
    pixel_count = {}
    png_list = glob.glob(os.path.join(path, "*.png"))
    for img in tqdm(png_list):
        img_array = cv2.imread(img)
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        pixel_count[Path(img).stem] = np.count_nonzero(img_gray)
    print(pixel_count)



if __name__ == "__main__":

    print(torch.cuda.is_available())


