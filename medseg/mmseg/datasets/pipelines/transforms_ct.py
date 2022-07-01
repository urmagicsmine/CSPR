import mmcv
import torch
import numpy as np
from mmcv.utils import deprecated_api_warning
from numpy import random
import torch.nn.functional as F

from ..builder import PIPELINES


@PIPELINES.register_module()
class NormZData(object):
    """normalize volumetric data & seg in the z-axis to specified slice spacing.
       data in (h, w, d) format

    This transform resize the z-axis of input image to specified slice spacing(or normz_scale).
    If the input dict contains the key "normz_scale", then the normz_scale in the input dict is used,
    otherwise the specified normz_scale in the init method is used. Note only normz_scale[0] is used 
    as the target depth, width and height are original img-shape.

    ``target_slice_spacings`` can either be a float (single-scale) or a list of float
    (multi-scale).

    ``target_slice_spacing``, ``slice_spacing_idx``, ``normz_scale`` and ``slice_scale_factor`` will be added.
    ``normz_scale``, ``img_shape``, ``pad_shape`` will be updated

    ``normz_scale``:         target ct shape, normz_scale[2]==depth, like ``scale`` in 2D resize 
    ``img_shape``:           update img_shape, which is the same as normz_scale
    ``slice_scale_factor``:  factor for depth resize, compared to previous depth

    Args:
        target_slice_spacing (float or list[float]): slice spacings for normalization.
                                                     Keep the original data if set to None.
    """

    def __init__(self,
                 target_slice_spacings=None):
         
        if target_slice_spacings is None:
            self.target_slice_spacings = None
        else:
            if isinstance(target_slice_spacings, list):
                self.target_slice_spacings = target_slice_spacings
            else:
                self.target_slice_spacings = [target_slice_spacings]

    def _random_scale(self, results):
        """Randomly sample a target_slice_spacing.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'target_slice_spacing` and 'slice_spacing_idx` are added into
                ``results``, which would be used by subsequent pipelines.
        """
        if self.target_slice_spacings is None:
            results['target_slice_spacing'] = results['spacing_list'][2]
            results['slice_spacing_idx'] = None
        else:
            slice_spacing_idx = np.random.randint(len(self.target_slice_spacings))
            target_slice_spacing = self.target_slice_spacings[slice_spacing_idx]
            results['target_slice_spacing'] = target_slice_spacing
            results['slice_spacing_idx'] = slice_spacing_idx

        target_depth = self.get_target_depth(results)
        #Only depth will be changed in this transform function.
        target_shape = [results['img_shape'][0], results['img_shape'][1], target_depth]
        results['normz_scale'] = target_shape

    @staticmethod
    def do_normalization(volume_data, target_shape=None, mode='nearest'):
        """Normalize zaxis and return uint8 array """
        img_h, img_w, img_d = volume_data.shape
        volume_data = torch.Tensor(np.float32(volume_data))
        volume_data = volume_data.view(1, 1, img_h, img_w, img_d)
        if mode == 'nearest':
            resize_tensor = F.interpolate(volume_data, size=target_shape, mode=mode).data[0, 0]
        else:
            resize_tensor = F.interpolate(volume_data, size=target_shape, mode=mode, align_corners=False).data[0, 0]
        resize_tensor = np.uint8(resize_tensor.clamp(0, 255).cpu().numpy())     
        return resize_tensor 

    @staticmethod
    def get_target_depth(results):
        # original depth and spacing list[2] has correspondence.
        target_slice_spacing = results['target_slice_spacing'] if results['target_slice_spacing'] is not None else results['spacing_list'][2]
        return int(round(results['ori_shape'][2] * results['spacing_list'][2] / target_slice_spacing))
 
    def _normz_img(self, results):
        """Resize images with ``results['target_slice_spacing']``."""
        if results['spacing_list'][2] == results['target_slice_spacing']:
            resize_tensor = np.uint8(results['img'])
            d_scale_factor = np.array([1.0], dtype=np.float32)
        else:
            resize_tensor = self.do_normalization(results['img'], target_shape=results['normz_scale'], mode='trilinear')
            new_d = resize_tensor.shape[2]
            d = results['img'].shape[2]
            d_scale_factor = np.array([new_d / d], dtype=np.float32)

        results['img'] = resize_tensor
        results['img_shape'] = resize_tensor.shape
        results['pad_shape'] = resize_tensor.shape
        results['slice_scale_factor'] = d_scale_factor

    def _normz_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            if results['normz_scale'] == results[key].shape:
                gt_seg = np.uint8(results[key])
            else:
                gt_seg = self.do_normalization(results[key], target_shape=results['normz_scale'], mode='nearest')
            results['gt_semantic_seg'] = gt_seg

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor',
                'keep_ratio' keys are added into result dict.
        """

        if 'target_slice_spacing' not in results:
            self._random_scale(results)
        else:
            # For test mode, fill ``normz_scale`` with predefined ``target_slice_spacing``
            target_depth = self.get_target_depth(results)
            target_shape = [results['img_shape'][0], results['img_shape'][1], target_depth]
            results['normz_scale'] = target_shape

        self._normz_img(results)
        self._normz_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'target_slice_spacing={self.target_slice_spacing}'
        return repr_str


@PIPELINES.register_module()
class RandomZFlip(object):
    """Flip the volumetric data & seg in the z-direction

    If the input dict contains the key "zflip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        prob (float, optional): The flipping probability. Default: None.
    """

    @deprecated_api_warning({'flip_ratio': 'prob'}, cls_name='RandomZFlip')
    def __init__(self, prob=None):
        self.prob = prob
        if prob is not None:
            assert prob >= 0 and prob <= 1

    def __call__(self, results):
        """Call function to zflip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'zflip' keys are added into result dict.
        """

        if 'zflip' not in results:
            zflip = True if np.random.rand() < self.prob else False
            results['zflip'] = zflip
        if results['zflip']:
            # flip image
            results['img'] = results['img'][:, :, ::-1]

            # flip segs
            for key in results.get('seg_fields', []):
                # use copy() to make numpy stride positive
                results[key] = results[key][:, :, ::-1].copy()
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'


@PIPELINES.register_module()
class RandomZCrop(object):
    """Random crop the volumetric data & seg in the z-axis.

    Args:
        crop_depth (tuple): Expected size after cropping.
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
    """

    def __init__(self, crop_depth, cat_max_ratio=1., ignore_index=255):
        self.crop_depth = crop_depth
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def get_valid_z_range(self, label):
        """Get valid z-range where every slice has foreground label.
        Args: label: np.array with shape as (h, w, d)
        """
        flatten = np.sum(np.sum(label, axis=0), axis=0)
        flags = np.where(flatten > 0)[0]
        valid_range = min(flags), max(flags)
        return valid_range

    def get_crop_bbox(self, img, valid_range=None):
        """Randomly get a crop bounding box(interval)."""
        if valid_range:
            margin_d = max(valid_range[1] - self.crop_depth, 0)
            offset_d = np.random.randint(valid_range[0], margin_d + 1)
        else:
            margin_d = max(img.shape[2] - self.crop_depth, 0)
            offset_d = np.random.randint(0, margin_d + 1)
        crop_d1, crop_d2 = offset_d, offset_d + self.crop_depth

        return crop_d1, crop_d2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_d1, crop_d2 = crop_bbox
        img = img[:, :, crop_d1:crop_d2]
        return img

    def __call__(self, results):
        """Call function to randomly crop images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        img = results['img']
        valid_range = self.get_valid_z_range(results['gt_semantic_seg'])
        crop_bbox = self.get_crop_bbox(img, valid_range)
        if self.cat_max_ratio < 1.:
            # Repeat 10 times
            for _ in range(10):
                seg_temp = self.crop(results['gt_semantic_seg'], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(
                        cnt) < self.cat_max_ratio:
                    break
                crop_bbox = self.get_crop_bbox(img)

        # crop the image
        img = self.crop(img, crop_bbox)
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_bbox)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_depth={self.crop_depth})'


@PIPELINES.register_module()
class RandomCenterCrop(object):
    """Random center-crop the volumetric data & seg. Suppose in (h, w, d)

    Args:
        crop_size (tuple): Expected size after cropping.
        move (float): The maximum pixel range for center_crop
    """
    def __init__(self, crop_size, move=5):
        self.crop_size = crop_size
        self.move = move

    def center_crop(self, crop_center, img):
        y, x, z = crop_center
        h, w, d = self.crop_size
        cropped = img[y - h // 2:y + h // 2,
                      x - w // 2:x + w // 2,
                      z - d // 2:z + d // 2]
        return cropped
    
    def random_center(self, img):
        img_shape = img.shape
        offset = np.random.randint(-self.move, self.move + 1, size=3)
        crop_center = np.array(img_shape) // 2 + offset
        return crop_center

    def __call__(self, results):
        """Call function to randomly center_crop images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly center cropped results, 'img_shape' , 'pad_shape' and 'ori_shape' key
                  in result dict is updated according to crop size.
        """

        img = results['img']
        crop_center = self.random_center(img)

        # crop the image
        img = self.center_crop(crop_center, img)
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape
        results['pad_shape'] = img_shape
        results['ori_shape'] = img_shape # Need to update ori_shape for random-center-crop testing.
        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = self.center_crop(crop_center, results[key])

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


@PIPELINES.register_module()
class RandomRotate3D(object):
    """Random rotate the volumetric data & seg in all three dims.
    """
    # TODO: this is the acs imp for lidc seg. Double Check the implementation
    def __init__(self):
        pass

    def random_rotate(self, array, angle):
        '''using Euler angles method.
            angle: 0: no rotation, 1: rotate 90 deg, 2: rotate 180 deg, 3: rotate 270 deg
        '''
        #
        X = np.rot90(array, angle[0], axes=(0, 2))  # rotate in X-axis
        Y = np.rot90(X, angle[1], axes=(1, 2))  # rotate in Y'-axis
        Z = np.rot90(Y, angle[2], axes=(0, 1))  # rotate in Z"-axis
        return Z

    def __call__(self, results):
        """Call function to randomly rotate images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly rotated results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        angle = np.random.randint(4, size=3)
        img = results['img']
        img = self.random_rotate(img, angle)

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = self.random_rotate(results[key], angle)

        return results
