import torch
import torch.nn.functional as F
import numpy as np
import pdb
import cv2
import os
import shutil

class SaveValues:
    def __init__(self, m):
        # register a hook to save values of activations and gradients
        self.activations = None
        self.gradients = None
        self.forward_hook = m.register_forward_hook(self.hook_fn_act)
        self.backward_hook = m.register_backward_hook(self.hook_fn_grad)

    def hook_fn_act(self, module, input, output):
        self.activations = output

    def hook_fn_grad(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


class CAM():
    def __init__(self, target_layer, weight_fc):
        """
        Args:
            target_layer: conv_layer before Global Average Pooling
            weight_fc: the weights of linear layer
        """

        self.target_layer = target_layer
        self.weight_fc = weight_fc

        # save values of activations and gradients in target_layer
        self.values = SaveValues(self.target_layer)

    def getCAM(self, size=None):
        """
        values: the activations and gradients of target_layer
            activations: feature map before GAP.  shape => (N, C, T, H, W)
        weight_fc: the weight of fully connected layer.  shape => (num_classes, C)
        cam: class activation map.  shape => (N, num_classes, T, H, W)
        """
        print(self.values.activations.shape)
        cam = F.conv3d(self.values.activations, weight=self.weight_fc[:, :, None, None, None])
        if size:
            cam = F.interpolate(cam, size=size, mode="trilinear", align_corners=False)
        return cam

    # def getCAM(self, size):
    #     """
    #     values: the activations and gradients of target_layer
    #         activations: feature map before GAP.  shape => (N, C, T, H, W)
    #     weight_fc: the weight of fully connected layer.  shape => (num_classes, C)
    #     size: shape of origin img => d,h,w
    #     cam: class activation map.  shape => (N, num_classes, T, H, W)
    #     """
    #
    #     b, _, d, h, w  = self.values.activations.shape[:]
    #     cls_num, channel_num = self.weight_fc.shape[:]
    #     cam = torch.zeros((b, cls_num, size[0], size[1], size[2]))
    #     for b_ in range(b):
    #         for c_ in range(cls_num):
    #             pdb.set_trace()
    #             feature_map = self.values.activations[b_, ...] # channel,d,h,w
    #             weight = self.weight_fc[c_]
    #             cam_b_c = torch.mm(weight.view(1, channel_num), feature_map.view(channel_num, -1))
    #             # cam_b_c = feature_map
    #             cam_b_c = cam_b_c.view(1, 1, d, h, w)
    #
    #             cam_b_c -= torch.min(cam_b_c)
    #             cam_b_c /= torch.max(cam_b_c)
    #             cam_b_c = F.interpolate(cam_b_c, size=size, mode="trilinear", align_corners=False)
    #             cam[b_, c_, ...] = cam_b_c[:, :, :]
    #
    #             # pdb.set_trace()
    #
    #     return cam
