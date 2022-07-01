import os.path as osp
import numpy as np
import tempfile
import SimpleITK as sitk

import mmcv
from mmcv.utils import print_log
from .builder import DATASETS
from .liver import LiverDataset, NiiLiverDataset

@DATASETS.register_module()
class CovidDataset(LiverDataset):

    CLASSES = ('background', 'lesion')

    PALETTE = [[120, 120, 120], [6, 230, 230]]

@DATASETS.register_module()
class NiiCovidDataset(NiiLiverDataset):

    CLASSES = ('background', 'leison')

    PALETTE = [[120, 120, 120], [6, 230, 230]]

