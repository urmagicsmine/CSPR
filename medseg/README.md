<div align="center">
  <img src="resources/mmseg-logo.png" width="600"/>
</div>
<br />

[![PyPI](https://img.shields.io/pypi/v/mmsegmentation)](https://pypi.org/project/mmsegmentation)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmsegmentation.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmsegmentation/workflows/build/badge.svg)](https://github.com/open-mmlab/mmsegmentation/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmsegmentation/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmsegmentation)
[![license](https://img.shields.io/github/license/open-mmlab/mmsegmentation.svg)](https://github.com/open-mmlab/mmsegmentation/blob/master/LICENSE)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmsegmentation.svg)](https://github.com/open-mmlab/mmsegmentation/issues)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmsegmentation.svg)](https://github.com/open-mmlab/mmsegmentation/issues)

Documentation: https://mmsegmentation.readthedocs.io/

## Introduction

MMSegmentation is an open source semantic segmentation toolbox based on PyTorch.
It is a part of the OpenMMLab project.

The master branch works with **PyTorch 1.3 to 1.6**.

![demo image](resources/seg_demo.gif)

### Major features

- **Unified Benchmark**

  We provide a unified benchmark toolbox for various semantic segmentation methods.

- **Modular Design**

  We decompose the semantic segmentation framework into different components and one can easily construct a customized semantic segmentation framework by combining different modules.

- **Support of multiple methods out of box**

  The toolbox directly supports popular and contemporary semantic segmentation frameworks, *e.g.* PSPNet, DeepLabV3, PSANet, DeepLabV3+, etc.

- **High efficiency**

  The training speed is faster than or comparable to other codebases.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Changelog

v0.7.0 was released in 07/10/2020.
Please refer to [changelog.md](docs/changelog.md) for details and release history.

## Benchmark and model zoo

Results and models are available in the [model zoo](docs/model_zoo.md).

Supported backbones:

- [x] ResNet
- [x] ResNeXt
- [x] [HRNet](configs/hrnet/README.md)
- [x] [ResNeSt](configs/resnest/README.md)
- [x] [MobileNetV2](configs/mobilenet_v2/README.md)

Supported methods:

- [x] [FCN](configs/fcn)
- [x] [PSPNet](configs/pspnet)
- [x] [DeepLabV3](configs/deeplabv3)
- [x] [PSANet](configs/psanet)
- [x] [DeepLabV3+](configs/deeplabv3plus)
- [x] [UPerNet](configs/upernet)
- [x] [NonLocal Net](configs/nonlocal_net)
- [x] [EncNet](configs/encnet)
- [x] [CCNet](configs/ccnet)
- [x] [DANet](configs/danet)
- [x] [GCNet](configs/gcnet)
- [x] [ANN](configs/ann)
- [x] [OCRNet](configs/ocrnet)
- [x] [Fast-SCNN](configs/fastscnn)
- [x] [Semantic FPN](configs/sem_fpn)
- [x] [PointRend](configs/point_rend)
- [x] [EMANet](configs/emanet)
- [x] [DNLNet](configs/dnlnet)
- [x] [CGNet](configs/cgnet)
- [x] [Mixed Precision (FP16) Training](configs/fp16/README.md)

## Installation

Please refer to [INSTALL.md](docs/install.md) for installation and dataset preparation.

## Get Started

Please see [getting_started.md](docs/getting_started.md) for the basic usage of MMSegmentation.
There are also tutorials for [adding new dataset](docs/tutorials/new_dataset.md), [designing data pipeline](docs/tutorials/data_pipeline.md), and [adding new modules](docs/tutorials/new_modules.md).

A Colab tutorial is also provided. You may preview the notebook [here](demo/MMSegmentation_Tutorial.ipynb) or directly [run](https://colab.research.google.com/github/open-mmlab/mmsegmentation/blob/master/demo/MMSegmentation_Tutorial.ipynb) on Colab.

## Contributing

We appreciate all contributions to improve MMSegmentation. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMSegmentation is an open source project that welcome any contribution and feedback.
We wish that the toolbox and benchmark could serve the growing research
community by providing a flexible as well as standardized toolkit to reimplement existing methods
and develop their own new semantic segmentation methods.

## Changes Made to MMCV

The naive resize function provided by mmcv is not compatible with mask of more than 3 channels, which is widely used for medical segmentation tasks. It should be replaced for correct behavior. The solution was provided here  https://github.com/open-mmlab/mmcv/issues/578

Add those functions to mmcv/image/geometric.py 


## Unsolved bugs
If set seg_pad_value to 255 in Pad for image processing pipelines, for input with depth >3, may encounter an assertion in opencv:

```cv2.error: OpenCV(4.4.0) /tmp/pip-req-build-cg56sov1/opencv/modules/core/src/copy.cpp:1463: error: (-215:Assertion failed) value[0] == value[1] && value[0] == value[2] && value[0] == value[3] in function 'copyMakeBorder'```