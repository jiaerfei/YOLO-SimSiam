# YOLO-SimSiam
## A self-supervised learning-based method for tunnel leakage defect few-shot recognition and visual explanation

![workflow](https://github.com/user-attachments/assets/986b768e-2183-4a83-8b87-5417137f3207)



### 1. Installation
参考ultralytics, Grad-CAM等 (https://github.com/ultralytics/ultralytics) (https://github.com/jacobgil/pytorch-grad-cam) 安装所需的各类第三方库


### 2. 代码说明

#### SSL_main.py
用于实现图像的特征学习

#### visualize.py
用于可视化各类结果，包含用于对比学习的图像，对比特征图，分割特征图等

#### box_main.py
用于正常训练和少样本训练

#### utils.py
用于数据预处理，标签格式转换等

其他为模型、数据集等配置文件


### 3. 数据集下载
其中包含了自监督、正常训练、kNN各任务数据集 (https://pan.baidu.com/s/1yahRkw1d4MVjj7Un4g_w_w)
