import argparse
import os
import cv2

import numpy as np
from ultralytics import YOLO
from ultralytics.utils.ops import scale_image
from ultralytics.utils.plotting import Annotator

from SSL_main import YOLOSimSiam

from utils import resize_to_show


parser = argparse.ArgumentParser("YOLO detection training only for GPUs")
# data
parser.add_argument("--data", default=r"C:\jiafeidl\YOLO_model_local\cfg\rock_data.yaml",
                    type=str, help="data root yaml path")
parser.add_argument("--batch-size", default=4, type=int, help="img nums for all gpus")
parser.add_argument("--input-size", default=640, type=int, help="image size for training")
# model
parser.add_argument("--pretrained", default="/home/featurize/work/YOLO_model/knn_contra/0.01_SGD_cos/weights/epoch200.pt",
                    type=str, help="pretrained checkpoint path")
parser.add_argument("--frozen", default=9, type=int, help="freeze n layers (backbone) to train")
# train
parser.add_argument("--epochs", default=100, type=int, help="train epoch nums")
parser.add_argument("--patience", default=0, type=float,
                    help="Number of epochs to wait without improvement in validation metrics before early stopping the training.")
parser.add_argument("--project", default="test", type=str, help="project name")
parser.add_argument("--name", default="test_loss", type=str,
                    help="creating a subdirectory within project folder, training logs and outputs are stored.")

# 其他可自定义设置的参数
# optimizer
parser.add_argument("--optim", default="Adam", type=str, help="optimize method")
parser.add_argument("--lr0", default=0.01, type=str, help="initial learning rate")
parser.add_argument("--lrf", default=0.01, type=str, help="initial learning rate")
# scheduler
parser.add_argument("--cos-lr", default=True, type=bool, help="whether to use cosine lr decay")


if __name__ == "__main__":
    ps = parser.parse_args()

    # normal train
    # model = YOLO(model="yolo11s-seg.yaml", task="segment", verbose=True)
    # train_result = model.train(data=ps.data, epochs=ps.epochs, imgsz=ps.input_size, batch=ps.batch_size,
    #                            name=ps.name, project=ps.project, patience=ps.patience,
    #                            optimizer=ps.optim, lr0=ps.lr0, cos_lr=ps.cos_lr, lrf=ps.lrf)


    # few train
    # model = YOLO(model="yolo11s-seg.yaml", task="segment", verbose=True).load(ps.pretrained)
    # train_result = model.train(data=ps.data, epochs=ps.epochs, imgsz=ps.input_size, batch=ps.batch_size,
    #                            name=ps.name, project=ps.project, patience=ps.patience,
    #                            optimizer=ps.optim, lr0=ps.lr0, cos_lr=ps.cos_lr, lrf=ps.lrf,
    #                            freeze=ps.frozen)
