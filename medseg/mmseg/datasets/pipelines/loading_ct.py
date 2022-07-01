import os.path as osp

import mmcv
import numpy as np

from ..builder import PIPELINES
from .load_utils import load_image_nii, adjust_ww_wl, load_npz, itk_load_nii 

@PIPELINES.register_module()
class LoadVolumeFromFile(object):
    """Load volumetric 3D data from file. Loaded data in (h, w, d) format

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0), "img_norm_cfg" (means=0 and stds=1) and 'spacing_list'

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
    """

    def __init__(self,
                 to_float32=False,
                 ct_type='nii',
                 window_width=400,
                 window_center=40):
        self.to_float32 = to_float32
        self.ct_type = ct_type
        self.window_width = window_width
        self.window_center = window_center

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        # volume_data is in (h, w, d) format, same as default mmcv output
        if self.ct_type=='nii':
            volume_data, spacing_list = itk_load_nii(filename)
            #volume_data, spacing_list = load_image_nii(filename)
        elif self.ct_type == 'npz':
            volume_data, spacing_list = load_npz(filename)
        else:
            raise NotImplementedError
        volume_data = adjust_ww_wl(volume_data, ww=self.window_width, wc=self.window_center, is_uint8=True) 
        if self.to_float32:
            volume_data = volume_data.astype(np.float32)
        results['spacing_list'] = spacing_list
        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = volume_data
        results['img_shape'] = volume_data.shape
        results['ori_shape'] = volume_data.shape
        # Decompose ct shape to [depth] and [height, width]
        #results['depth'] = volume_data.shape[2]
        #results['ori_depth'] = volume_data.shape[2]
        # Set initial values for default meta_keys
        results['pad_shape'] = volume_data.shape
        results['scale_factor'] = 1.0
        num_channels = 1 
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"ct_type='{self.ct_type}',"
        repr_str += f"window_width='{self.window_width}',"
        repr_str += f"window_center='{self.window_center}',"
        return repr_str


@PIPELINES.register_module()
class LoadVolumeAnnotations(object):
    """Load volume annotations for semantic segmentation. Loaded data in (h, w, d) format

    Args:
        reduct_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 ct_type="nii"):
        self.reduce_zero_label = reduce_zero_label
        self.ct_type = ct_type

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        if self.ct_type=='nii':
            gt_semantic_seg, _ = itk_load_nii(filename)
            #gt_semantic_seg, _ = load_image_nii(filename)
        else:
            raise NotImplementedError
        # modify if custom classes
        if results.get('label_map', None) is not None:
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg == old_id] = new_id
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"ct_type='{self.ct_type}',"
        return repr_str

@PIPELINES.register_module()
class LoadPairDataFromFile(object):
    """Load volumetric 3D data and seg from file. Loaded data in (h, w, d) format

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0), "img_norm_cfg" (means=0 and stds=1) and 'spacing_list'

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
    """

    def __init__(self,
                 to_float32=False,
                 ct_type='npz',
                 reduce_zero_label=False):
        self.to_float32 = to_float32
        self.ct_type = ct_type
        self.reduce_zero_label = reduce_zero_label

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        # volume_data is in (h, w, d) format, same as default mmcv output
        if self.ct_type == 'npz':
            with np.load(filename) as npz:
                volume_data, gt_semantic_seg = npz['voxel'], npz['answer1']
        else:
            raise NotImplementedError
        if self.to_float32:
            volume_data = volume_data.astype(np.float32)
        # modify if custom classes
        if results.get('label_map', None) is not None:
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg == old_id] = new_id
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = volume_data
        results['img_shape'] = volume_data.shape
        results['ori_shape'] = volume_data.shape
        # Decompose ct shape to [depth] and [height, width]
        #results['depth'] = volume_data.shape[2]
        #results['ori_depth'] = volume_data.shape[2]
        # Set initial values for default meta_keys
        results['pad_shape'] = volume_data.shape
        results['scale_factor'] = 1.0
        num_channels = 1
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"ct_type='{self.ct_type}',"
        return repr_str
