import mmcv
import numpy as np
from mmcv.utils import deprecated_api_warning
from numpy import random
import albumentations
from albumentations import Compose
import cv2
from ..builder import PIPELINES


# LJ
@PIPELINES.register_module()
class TensorResize(object):
    """Resize images.

    This transform resizes the input image to some scale. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 2 multiscale modes:
    - `ratio_range` is None and `multiscale_mode` == "range": randomly sample a
        scale from the a range.
    - `ratio_range` is None and `multiscale_mode` == "value": randomly sample a
        scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range'):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)


        # given multiple scales or a range of scales
        assert multiscale_mode in ['value', 'range']
        self.multiscale_mode = multiscale_mode

    @staticmethod
    def random_select(img_scales):
        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        edges = [s[0] for s in img_scales]
        scale_edge = np.random.randint(
            min(edges),
            max(edges) + 1)
        img_scale = (scale_edge, scale_edge)
        return img_scale, None


    def _random_scale(self, results):
        if len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):

        img, w_scale, h_scale = self._imresize(
            results['img'], results['scale'], return_scale=True)
        scale_factor = np.array([w_scale, h_scale], dtype=np.float32)

        results['img'] = img
        results['img_shape'] = img.shape
        results['scale_factor'] = scale_factor

    def _resize_seg(self, results):
        gt_semantic_seg, w_scale, h_scale = self._imresize(
            results['gt_semantic_seg'], results['scale'], return_scale=True)

        results['gt_semantic_seg'] = gt_semantic_seg

    def _imresize(self, img, size, return_scale=False):
        h, w, depth = img.shape[:]
        resized_img = []
        for d in range(depth):
            _img = cv2.resize(
                img[:, :, d], size, interpolation=cv2.INTER_LINEAR)

            resized_img.append(_img)

        # ForkedPdb().set_trace()
        resized_img = np.dstack(resized_img)

        if not return_scale:
            return resized_img
        else:
            w_scale = size[0] / w
            h_scale = size[1] / h
            return resized_img, w_scale, h_scale

    def __call__(self, results):
        # vis(results, tag='origin')

        if 'scale' not in results:
            self._random_scale(results)
        self._resize_img(results)
        if 'gt_semantic_seg' in results:
            self._resize_seg(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(img_scale={}, multiscale_mode={})'
                        ).format(self.img_scale, self.multiscale_mode)

        return repr_str


@PIPELINES.register_module()
class TensorXYCrop:
    def __init__(self, crop_size, move=5, train=True):
        self.size = (crop_size, ) * 2 if isinstance(crop_size, int) else crop_size
        self.move = move
        self.train = train

    def crop_xy(self, array, center, size):
        # For input tensor (h, w, d), crop along xy axis

        y, x, z = center
        h, w = size
        cropped = array[y - h // 2:y + h // 2,
                  x - w // 2:x + w // 2,
                  ...]

        return cropped

    def random_center(self, shape, move):
        offset = np.random.randint(-move, move + 1, size=3)
        yxz = np.array(shape) // 2 + offset
        return yxz

    def get_crop_center(self, voxel):
        shape = voxel.shape
        # norm
        if self.train:
            if self.move is not None:
                center = self.random_center(shape, self.move)
            else:
                center = np.array(shape) // 2
        else:
            center = np.array(shape) // 2

        return center


    def _crop_img(self, results, center):

        results['img'] = self.crop_xy(results['img'], center, self.size)


    def _crop_seg(self, results, center):
        results['gt_semantic_seg'] = self.crop_xy(results['gt_semantic_seg'], center, self.size)

    def __call__(self, results):

        center = self.get_crop_center(results['img'])
        self._crop_img(results, center)
        if 'gt_semantic_seg' in results:
            self._crop_seg(results, center)

            if 'gt_label' in results:
                # crop后判断是否更新gt_lable np.sum()费时?
                _label = 0 if np.sum(results['gt_semantic_seg']) == 0 else 1
                results['gt_label'] = np.array(_label, dtype=np.int64)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(Crop_Transform)'
        return repr_str


@PIPELINES.register_module()
class TensorMultiZCrop:
    def __init__(self, num_slice, crop_num=0):
        self.crop_num = crop_num
        self.num_slice = num_slice

    def crop_z(self, array, z_start, z_end):
        # For input tensor (h, w, d), crop along xy axis

        cropped = array[..., z_start:z_end]

        return cropped

    def get_params(self, voxel):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (ndarray): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: Params (xmin, ymin, target_height, target_width) to be
                passed to ``crop`` for random crop.
        """
        h, w, depth = voxel.shape[:]
        assert depth >= self.num_slice
        move_z = self.crop_num
        stride_z = (depth - self.num_slice) // (move_z - 1)

        params = []
        z_start = -stride_z
        for i in range(move_z - 1):
            z_start = z_start + stride_z
            z_end = z_start + self.num_slice
            params.append([z_start, z_end])

        # 尾部对齐
        params.append([depth - self.num_slice, depth])
        assert len(params) == self.crop_num

        return params

    def __call__(self, results):
        voxel = results['img']

        voxel_ret = []
        params = self.get_params(voxel)
        assert isinstance(params, list)
        for i in range(len(params)):
            z_start, z_end = params[i]
            voxel_ = self.crop_z(voxel, z_start, z_end)
            voxel_ret.append(voxel_)

        voxel_ret = np.stack(voxel_ret)

        results['img'] = voxel_ret

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(ZCrop_Transform)'
        return repr_str


@PIPELINES.register_module()
class TensorNorm:
    def __init__(self, mean=99., std=77.):
        self.mean = mean
        self.std = std

    def __call__(self, results):

        voxel = results['img']

        voxel = (voxel - self.mean) / self.std
        if len(voxel.shape) < 4:
            voxel = np.expand_dims(voxel, 0).astype(np.float32)

        results['img'] = voxel

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(Norm_Transform)'
        return repr_str
