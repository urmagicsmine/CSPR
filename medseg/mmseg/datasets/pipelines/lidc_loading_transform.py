import numpy as np
import torch
import os
from scipy import ndimage
from skimage import measure, morphology

from ..builder import PIPELINES

@PIPELINES.register_module()
class LoadTensorFromFile(object):
    """Load 3d tensors from npz file.
        (noted by lzh) : LIDC-IDRI is supported for the time being.
    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes()`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        filename = results['img_info']['filename']
        with np.load(filename) as npz:
            img, seg = npz['voxel'], npz['answer1']
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        results['img'] = img
        results['ann'] = seg
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['pad_shape'] = (0,) * 3
        # key'gt_semantic_seg' is used for transform
        results['gt_semantic_seg'] = seg
        results['seg_fields'].append('gt_semantic_seg')
        #num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        num_channels = 1
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str

@PIPELINES.register_module()
class TensorNormCropRotateFlip:
    def __init__(self, size, move=5, mean=99.2565, std=77.3798,
            train=True, copy_channels=False, reduce_zero_label=False):
        self.size = (size, )*3
        self.move = move
        self.mean = mean
        self.std = std
        self.copy_channels = copy_channels
        self.reduce_zero_label = reduce_zero_label
        self.train = train

    def __call__(self, results):
        # results.keys():  dict_keys(['img_info', 'seg_fields', 'filename', 'img', 'ann', \
        #                  'img_shape', 'ori_shape', 'img_norm_cfg'])
        for key in results.get('img_fields', ['img']):
            voxel = results[key]
        seg = results['ann']
        shape = voxel.shape
        voxel = (voxel - self.mean) / self.std
        #voxel = voxel/255. - 1 # original norm in ACS
        if self.train:
            if self.move is not None:
                center = random_center(shape, self.move)
            else:
                center = np.array(shape) // 2
            voxel_ret = crop(voxel, center, self.size)
            seg_ret = crop(seg, center, self.size)
            angle = np.random.randint(4, size=3)
            voxel_ret = rotation(voxel_ret, angle=angle)
            seg_ret = rotation(seg_ret, angle=angle)
            axis = np.random.randint(4) - 1
            voxel_ret = reflection(voxel_ret, axis=axis)
            seg_ret = reflection(seg_ret, axis=axis)
        else:
            center = np.array(shape) // 2
            voxel_ret = crop(voxel, center, self.size)
            seg_ret = crop(seg, center, self.size)
        results['img_shape'] = voxel_ret.shape
        results['ori_shape'] = voxel_ret.shape
        results['pad_shape'] = (0,) * 3
        #if self.copy_channels:
            #img = np.stack([voxel_ret,voxel_ret,voxel_ret],0).astype(np.float32)
            #ann = np.expand_dims(seg_ret,0).astype(np.float32)
            #ann = seg_ret
        #else:
            #img = np.expand_dims(voxel_ret, 0).astype(np.float32)
            #ann = np.expand_dims(seg_ret,0).astype(np.float32)
        img = voxel_ret.astype(np.float32)
        ann = seg_ret.astype(np.float32)
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['img'] = img
        #results['seg_fields'] = ann
        results['gt_semantic_seg'] = ann
        return results

def resize(voxel, spacing, new_spacing=[1., 1., 1.]):
    '''Resize `voxel` from `spacing` to `new_spacing`.'''
    resize_factor = []
    for sp, nsp in zip(spacing, new_spacing):
        resize_factor.append(float(sp) / nsp)
    resized = ndimage.interpolation.zoom(
        voxel, resize_factor, mode='nearest')
    for i, (sp, shape, rshape) in enumerate(zip(spacing, voxel.shape, resized.shape)):
        new_spacing[i] = float(sp) * shape / rshape
    return resized, new_spacing

def rotation(array, angle):
    '''using Euler angles method.
        angle: 0: no rotation, 1: rotate 90 deg, 2: rotate 180 deg, 3: rotate 270 deg
    '''
    #
    X = np.rot90(array, angle[0], axes=(0, 1))  # rotate in X-axis
    Y = np.rot90(X, angle[1], axes=(0, 2))  # rotate in Y'-axis
    Z = np.rot90(Y, angle[2], axes=(1, 2))  # rotate in Z"-axis
    return Z


def reflection(array, axis):
    '''
        axis: -1: no flip, 0: Z-axis, 1: Y-axis, 2: X-axis
    '''
    if axis != -1:
        ref = np.flip(array, axis)
    else:
        ref = np.copy(array)
    return ref


def crop(array, zyx, dhw):
    z, y, x = zyx
    d, h, w = dhw
    cropped = array[z - d // 2:z + d // 2,
              y - h // 2:y + h // 2,
              x - w // 2:x + w // 2]
    return cropped

def random_center(shape, move):
    offset = np.random.randint(-move, move + 1, size=3)
    zyx = np.array(shape) // 2 + offset
    return zyx
