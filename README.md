# Advancing 3D Medical Image Analysis with Variable Dimension Transform based Supervised 3D Pre-training

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/advancing-3d-medical-image-analysis-with/medical-object-detection-on-deeplesion)](https://paperswithcode.com/sota/medical-object-detection-on-deeplesion?p=advancing-3d-medical-image-analysis-with)

## Introduction
This is an implementation of our paper **Advancing 3D Medical Image Analysis with Variable Dimension Transform based Supervised 3D Pre-training**.

Modified from [mmclassification](https://github.com/open-mmlab/mmclassification).

Support **3D** ResNet pre-training with **2D** natural-image dataset.


## Installation

Please refer to [install.md](docs/install.md) for installation.

## Data preparation
Download ImageNet dataset and put it as the following structure:

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

The pre-trained 3D-ResNet-18 model can be downloaded from [BaiduYun](https://pan.baidu.com/s/1dUv-YCv_FL02ywOxk-blqw)(verification code: 865y) or [GoogleDrive](https://drive.google.com/file/d/1TlaGFA154RfunoLFzvP8SRjzVL8zGTOI/view?usp=share_link). 

<!--Make a folder named ```pretrained_model``` and put pre-trained models in it.-->

We also provide pretrained models of other compared methods for convenience, which can be avaliable at [BaiduYun](https://pan.baidu.com/s/1YYlKcL4wAwoIeIJKxPq1Dg)(verification code: 1ezg) or [GoogleDrive](https://drive.google.com/drive/folders/1yiVxKtCOkNF9mPcRrYNguetu0YQLJ8lq?usp=sharing). Note that the first convolution layer of some pretrained models are with input channels of 3(same as imagenet pretrained models), which is inconsistent with 1-channel CT slice input. To fully use the pretrained weights, we averaged the parameters of the first conv layer so that it can take 1-channel input data. These models are end with suffixes like ```_med1channel.pth```.

## Transfer to other tasks

### Classification
LIDC-Classification

Data split can be found in data/lidc.
```
bash tools/dist_train.sh configs/LIDC_Cls_configs/resnet18_3d_BN_b32x4_LIDC_cos_smoo.py $NUM_GPUS
```

### Segmentation

Data split can be found in medseg/data.

We provide a implementation modified from [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) for segmentation experiments on BCV, LIDC and LiTS datasets. Please run corresponding bash files under ```medseg``` folder.
```
medseg/bcv_test.sh
medseg/bcv_train.sh
medseg/lidc_test.sh
medseg/lidc_train.sh
medseg/LiTS_test.sh
medseg/LiTS_train.sh
```

### Detection 
Please refer to our another [repo](https://github.com/urmagicsmine/MP3D) to run experiments on the DeepLesion dataset. To train a P3D63 model, run:
```
bash tools/dist_train.sh configs/deeplesion/p3d.py 8
```

## Contact
If you have questions or suggestions, please open an issue here.

