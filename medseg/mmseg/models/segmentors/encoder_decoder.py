import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize, resize_3d
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor
from ..backbones.acsconv.converters import ConvertModel
import numpy as np


@SEGMENTORS.register_module()
class EncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 converter=None,
                 neck=None,
                 classifier_neck=None,
                 classifier_head=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 is_3d_pred=False):
        super(EncoderDecoder, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        # LJ
        if classifier_neck is not None:
            self.classifier_neck = builder.build_neck(classifier_neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)
        self._init_classifier_head(classifier_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.is_3d_pred = is_3d_pred

        self.init_weights(pretrained=pretrained)

        if converter is not None:
            # convert 2d model to 3d after init_weights()
            # optional: [ACS, I3D, 2.5D]
            print('Convert model with \'{}\' function'.format(converter))
            #self.backbone = ConvertModel(self.backbone, converter).model
            self = ConvertModel(self, converter).model
            #print(self)

        assert self.with_decode_head
        if self.is_3d_pred and converter is None:
            assert self.decode_head.conv_cfg is not None, "Decode head must use conv3d for 3d prediction"
            assert self.decode_head.conv_cfg['type'] == 'Conv3d', "Decode head must use conv3d for 3d prediction"
        else:
            if all(map(self.has_value, [self.backbone.conv_cfg, self.decode_head.conv_cfg])):
                print("===Warning: Change to 3D prediction to accomodate for Conv3d operations.===")
                self.is_3d_pred=True

    @staticmethod
    def has_value(input_dict, key_name='type', value_name='Conv3d'):
        if input_dict is None:
            return False
        elif input_dict[key_name] == value_name:
            return True

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def _init_classifier_head(self, classifier_head):
        """Initialize ``classifier_head``"""
        if classifier_head is not None:
            self.classifier_head = builder.build_head(classifier_head)


    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        super(EncoderDecoder, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()
        # LJ
        if self.with_classifier:
            self.classifier_head.init_weights()

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        if not self.is_3d_pred:
            out = resize(
                input=out,
                size=img.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        else:
            out = resize_3d(
                input=out,
                size=img.shape[-3:],
                mode='trilinear',
                align_corners=self.align_corners)
        return out

    # LJ
    def _classifier_head_forward_train(self, x, gt_label):
        losses = dict()
        loss_classifier = self.classifier_head.forward_train(x, gt_label)

        losses.update(add_prefix(loss_classifier, 'classifier'))

        return losses

    def _classifier_head_simple_test(self, img):
        x = self.backbone(img)
        if isinstance(x, tuple):
            x_ = self.classifier_neck(x[-1])
        else:
            x_ = self.classifier_neck(x)

        pred = self.classifier_head.simple_test(x_)

        return pred


    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def forward_train_wclassifier(self, img, img_metas, gt_semantic_seg, gt_label):

        x = self.backbone(img)

        losses = dict()

        if isinstance(x, tuple):
            x_ = self.classifier_neck(x[-1])
        else:
            x_ = self.classifier_neck(x)

        loss_classifier = self._classifier_head_forward_train(x_, gt_label)
        losses.update(loss_classifier)

        if self.with_neck:
            x = self.neck(x)

        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return losses


    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        ########### lzh debug ############
        ### these code can be used to calculate the percent of each class in gt_seg_map ###
        ##################################
        #from functools import reduce
        ##print('input shape, mean and std:', img.shape, torch.mean(img), torch.std(img))
        #print('unique label:', gt_semantic_seg.shape, torch.unique(gt_semantic_seg))
        #voxels = reduce(lambda x,y : x*y, gt_semantic_seg.shape)
        #unique_item = torch.unique(gt_semantic_seg)
        #for item in unique_item:
            #counts = torch.sum(gt_semantic_seg==item)
            #percent = counts*1.0 / voxels  * 100
            #print(item, counts, '\t%.3f' % percent)
            #print(item, counts, voxels,  percent)
        ################################

        x = self.extract_feat(img)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return losses

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            if not self.is_3d_pred:
                preds = resize(
                    preds,
                    size=img_meta[0]['ori_shape'][:2],
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False)
            else:
                preds = resize_3d(
                    preds,
                    size=(img_meta[0]['ori_shape'][2],) + img_meta[0]['ori_shape'][:2],
                    mode='trilinear',
                    align_corners=self.align_corners,
                    warning=False)
        return preds

    # TODO refactor
    def slide3d_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap for volumetric data.

        If d_crop > d_img or h_crop > h_img or w_crop > w_img, the small 
        patch will be used to decode without padding.
        """

        d_stride, h_stride, w_stride = self.test_cfg.stride
        d_crop, h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, d_img, h_img, w_img = img.size()
        num_classes = self.num_classes
        d_grids = max(d_img - d_crop + d_stride - 1, 0) // d_stride + 1
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, d_img, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, d_img, h_img, w_img))
        for d_idx in range(d_grids):
            for h_idx in range(h_grids):
                for w_idx in range(w_grids):
                    z1 = d_idx * d_stride
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    z2 = min(z1 + d_crop, d_img)
                    y2 = min(y1 + h_crop, h_img)
                    x2 = min(x1 + w_crop, w_img)
                    z1 = max(z2 - d_crop, 0)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)
                    crop_img = img[:, :, z1:z2, y1:y2, x1:x2]
                    crop_seg_logit = self.encode_decode(crop_img, img_meta)
                    preds += F.pad(crop_seg_logit,
                                   (int(x1), int(preds.shape[4] - x2), int(y1),
                                    int(preds.shape[3] - y2), int(z1), int(preds.shape[2] - z2)))
    
                    count_mat[:, :, z1:z2, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            if not self.is_3d_pred:
                preds = resize(
                    preds,
                    size=img_meta[0]['ori_shape'][:2],
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False)
            else:
                preds = resize_3d(
                    preds,
                    size=(img_meta[0]['ori_shape'][2],) + img_meta[0]['ori_shape'][:2],
                    mode='trilinear',
                    align_corners=self.align_corners,
                    warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            if not self.is_3d_pred:
                seg_logit = resize(
                    seg_logit,
                    size=img_meta[0]['ori_shape'][:2],
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False)
            else:
                seg_logit = resize_3d(
                    seg_logit,
                    size=(img_meta[0]['ori_shape'][2],) + img_meta[0]['ori_shape'][:2],
                    mode='trilinear',
                    align_corners=self.align_corners,
                    warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole', 'slide3d', 'slideZ']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        elif self.test_cfg.mode == 'slide3d':
            seg_logit = self.slide3d_inference(img, img_meta, rescale)
        elif self.test_cfg.mode == 'slideZ':
            if self.test_cfg.wclassifier:
                seg_logit, classifier_preds = self.slideZ_inference(img, img_meta, rescale)
            else:
                seg_logit = self.slideZ_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        #output = F.softmax(seg_logit, dim=1)
        if len(seg_logit.shape) == 4:
            output = F.softmax(seg_logit, dim=1)
        elif len(seg_logit.shape) == 5:
            # save GPU memory 
            # seg_pred = torch.ones([1] + list(seg_logit.shape[-3:]), dtype=torch.int64)
            for i in range(seg_logit.shape[-3]):
                seg_logit[:,:,i,:,:] = F.softmax(seg_logit[:,:,i,:,:], dim=1)
            output = seg_logit

        # LJ
        if 'flip' in img_meta[0]:
            flip = img_meta[0]['flip']

            if flip:
                flip_direction = img_meta[0]['flip_direction']
                assert flip_direction in ['horizontal', 'vertical']
                if flip_direction == 'horizontal':
                    output = output.flip(dims=(-1, ))
                elif flip_direction == 'vertical':
                    output = output.flip(dims=(-2, ))

        # LJ
        if 'zflip' in img_meta[0]:
            # Added for 3D prediction
            zflip = img_meta[0]['zflip']
            if zflip:
                output = output.flip(dims=(-3, ))

        # LJ
        if self.test_cfg.mode == 'slideZ' and self.test_cfg.wclassifier:
            return output, classifier_preds
        else:
            return output

    # LJ
    def slideZ_inference(self, img, img_meta, rescale):
        assert self.test_cfg.mode == 'slideZ'
        d_stride = self.test_cfg.stride
        num_slice = self.test_cfg.num_slice
        batch_size, _, d_img, h_img, w_img = img.size()
        assert batch_size == 1
        assert d_img >= num_slice
        num_classes = self.num_classes

        params = []
        z_start, z_end = 0, num_slice
        while z_end < d_img:
            params.append([z_start, z_end])
            z_start = z_start + d_stride
            z_end = z_start + num_slice
        # 尾部对齐
        params.append([d_img - num_slice, d_img])

        classifier_preds = []
        seg_preds = img.new_zeros((batch_size, num_classes, d_img, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, d_img, h_img, w_img))
        for idx in range(len(params)):
            z_start, z_end = params[idx]
            crop_img = img[:, :, z_start:z_end, ...]
            crop_seg_logit = self.encode_decode(crop_img, img_meta)
            seg_preds += F.pad(crop_seg_logit, (0, 0, 0, 0, int(z_start), int(seg_preds.shape[2] - z_end)))
            count_mat[:, :, z_start:z_end, ...] += 1
            if self.test_cfg.wclassifier:
                classifier_pred = self._classifier_head_simple_test(crop_img)
                classifier_preds.append(classifier_pred)

        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        seg_preds = seg_preds / count_mat

        if rescale:
            if not self.is_3d_pred:
                seg_preds = resize(
                    seg_preds,
                    size=img_meta[0]['ori_shape'][:2],
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False)
            else:
                seg_preds = resize_3d(
                    seg_preds,
                    size=(img_meta[0]['ori_shape'][2],) + img_meta[0]['ori_shape'][:2],
                    mode='trilinear',
                    align_corners=self.align_corners,
                    warning=False)

        if self.test_cfg.wclassifier:
            return seg_preds, classifier_preds
        else:
            return seg_preds

    # LJ
    def simple_test_wclassifier(self, img, img_meta, rescale=True):

        seg_logit, classifier_preds = self.inference(img, img_meta, rescale)

        if len(seg_logit.shape) == 4:
            seg_pred = seg_logit.argmax(dim=1)
        elif len(seg_logit.shape) == 5:
            # save GPU memory
            seg_pred = torch.ones([1] + list(seg_logit.shape[-3:]), dtype=torch.int64)
            for i in range(seg_logit.shape[-3]):
                seg_pred[:,i,:,:] = seg_logit[:,:,i,:,:].argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # 节省内存
        seg_pred = np.array(seg_pred, dtype=np.uint8)
        # unravel batch dim
        seg_pred = list(seg_pred)

        classifier_pred = np.stack(classifier_preds)
        classifier_pred = np.max(classifier_pred, axis=0)
        classifier_pred[:, 0] = 1 - classifier_pred[:, 1] + 1e-6
        classifier_pred = [np.array(p, dtype=np.float16) for p in classifier_pred]

        output = {}
        output['seg_pred'] = seg_pred
        output['classifier_pred'] = classifier_pred

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        if len(seg_logit.shape) == 4:
            seg_pred = seg_logit.argmax(dim=1)
        elif len(seg_logit.shape) == 5:
            # save GPU memory 
            seg_pred = torch.ones([1] + list(seg_logit.shape[-3:]), dtype=torch.int64)
            for i in range(seg_logit.shape[-3]):
                seg_pred[:,i,:,:] = seg_logit[:,:,i,:,:].argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # LJ 节省内存
        seg_pred = np.array(seg_pred, dtype=np.uint8)
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        if len(seg_logit.shape) == 4:
            seg_pred = seg_logit.argmax(dim=1)
        elif len(seg_logit.shape) == 5:
            # save GPU memory 
            seg_pred = torch.ones([1] + list(seg_logit.shape[-3:]), dtype=torch.int64)
            for i in range(seg_logit.shape[-3]):
                seg_pred[:,i,:,:] = seg_logit[:,:,i,:,:].argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
