## Introduction
This is an implementation of our paper **Supervised 3D Pre-training on Large-scale 2D Natural Image Datasets for 3D Medical Image Analysis**.

Modified from [mmclassification](https://github.com/open-mmlab/mmclassification).

Support **3D** ResNet pre-training with **2D** natural-image dataset.


## Installation

Please refer to [install.md](docs/install.md) for installation.

## Data preparation
Download ImageNet dataset and put as the following structure:

```
data
  ├──imagenet
        ├── get_meta.sh
        ├── meta
        │   ├── val.txt
        │   ├── test.txt
        │   ├── train.txt
        ├── val
        │   ├──ILSVRC2012_val_00000001.JPEG
        │   ├──ILSVRC2012_val_00000002.JPEG
        │   ├── ...
        ├── test
        │   ├──ILSVRC2012_test_00000001.JPEG
        │   ├──ILSVRC2012_test_00000002.JPEG
        │   ├── ...
        └── train
              └── n10148035
              │    ├── n10148035_10034.JPEG
              │    ├── n10148035_10371.JPEG
              │    ├── ...
              └── n11879895
              │    ├── ...
              └── ...
```





## Pre-train a 3D model on ImageNet dataset
Run this script to pre-train a 3D-ResNet-18 model on ImageNet dataset. It will take around 7 days on 8 Titan XP GPUs.
```
bash pre_train.sh
```

## Pre-trained Model
We provide models pre-trained on ImageNet dataset which can be used for different 3D medical image analysis tasks.

The pre-trained 3D-ResNet-18 model can be downloaded from [Google drive](https://drive.google.com/file/d/1vyyQvGxlffvOxZ8HpGcwJgrMeS00RWxW/view?usp=sharing) or [BaiduYun](https://pan.baidu.com/s/1Qk86bVLKLISUZ1L4FmqK6A)(verification code: 9hgg). 

<!--Make a folder named ```pretrained_model``` and put pre-trained models in it.-->


