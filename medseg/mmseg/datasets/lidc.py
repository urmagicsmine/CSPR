from torch.utils.data import Dataset
import random
import os
import os.path as osp
import numpy as np
import pandas as pd

import mmcv
from mmcv.utils import print_log
from mmseg.core import mean_dice

from .custom import CustomDataset
from .builder import DATASETS

@DATASETS.register_module()
class LIDCDataset(CustomDataset):
    CLASSES = ('background', 'lesion')
    PALETTE = [[120,120,120], [128, 64, 128]]

    def __init__(self, ann_file, size=48, *args, **kwargs):
        self.ann_file = ann_file
        self.size = (size, ) * 3
        self.disp_per_case = False
        super(LIDCDataset, self).__init__(*args, **kwargs)

    def load_annotations(self, *args, **kwargs):
        if not self.test_mode:
            self.names = pd.read_csv(os.path.join(self.data_root, self.ann_file))['train'].\
                dropna().map(lambda x: os.path.join(self.img_dir, x)).values
        else:
            self.names = pd.read_csv(os.path.join(self.data_root, self.ann_file))['test'].\
                dropna().map(lambda x: os.path.join(self.img_dir, x)).values
            #self.names = self.names[:200] # debug
        img_infos = [dict(filename=f_name) for f_name in self.names]
        return img_infos

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def get_gt_seg_maps(self):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        for img_info in self.img_infos:
            filename = img_info['filename']
            with np.load(filename) as npz:
                _, gt_seg_map = npz['voxel'], npz['answer1']
            gt_seg_map = gt_seg_map.astype(np.int64)
            # modify if custom classes
            if self.label_map is not None:
                for old_id, new_id in self.label_map.items():
                    gt_seg_map[gt_seg_map == old_id] = new_id
            if self.reduce_zero_label:
                # avoid using underflow conversion
                gt_seg_map[gt_seg_map == 0] = 255
                gt_seg_map = gt_seg_map - 1
                gt_seg_map[gt_seg_map == 254] = 255
            # center crop
            shape = gt_seg_map.shape
            center = np.array(shape) // 2
            gt_seg_map = crop(gt_seg_map, center, self.size)
            gt_seg_maps.append(gt_seg_map)

        return gt_seg_maps

    def evaluate(self, results, metric='mDice3D', logger=None, **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mIoU', 'mDice3D']
        if metric not in allowed_metrics:
            raise KeyError('metric {} is not supported'.format(metric))

        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps()
        gt_seg_maps = [gt_seg_map.transpose(2, 0, 1) for gt_seg_map in gt_seg_maps]
        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)
        all_acc, acc, dice, case_dice = mean_dice(results, gt_seg_maps, num_classes, ignore_index=self.ignore_index)
        summary_str = '\n'
        if self.disp_per_case:
            for ind in range(len(results)):
                summary_str += '%-25s:' % osp.basename(self.img_infos[ind]['filename']) + str(case_dice[ind]) + '\n'
            summary_str += "MeanDice per case: " + str(np.mean(case_dice, axis=0)) + '\n'
        #summary_str = ''
        summary_str += 'per class results:\n'

        line_format = '{:<15} {:>10} {:>10}\n'
        summary_str += line_format.format('Class', 'mDice3D', 'Acc')
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        remove_bg_item = lambda x: [x[ind] for ind in range(x.size) if class_names[ind] != 'background']
        acc, dice = list(map(remove_bg_item, [acc, dice]))
        class_names = [cls_name for cls_name in class_names if cls_name != 'background']

        for i in range(len(acc)):
            dice_str = '{:.2f}'.format(dice[i] * 100)
            acc_str = '{:.2f}'.format(acc[i] * 100)
            summary_str += line_format.format(class_names[i], dice_str, acc_str)
        summary_str += 'Summary:\n'
        line_format = '{:<15} {:>10} {:>10} {:>10}\n'
        summary_str += line_format.format('Scope', 'mDice3D', 'mAcc', 'aAcc')

        dice_str = '{:.2f}'.format(np.nanmean(dice) * 100)
        acc_str = '{:.2f}'.format(np.nanmean(acc) * 100)
        all_acc_str = '{:.2f}'.format(all_acc * 100)
        summary_str += line_format.format('global', dice_str, acc_str,
                                          all_acc_str)
        print_log(summary_str, logger)

        eval_results['mDice3D'] = np.nanmean(dice)
        eval_results['mAcc'] = np.nanmean(acc)
        eval_results['aAcc'] = all_acc

        return eval_results

def crop(array, zyx, dhw):
    z, y, x = zyx
    d, h, w = dhw
    cropped = array[z - d // 2:z + d // 2,
              y - h // 2:y + h // 2,
              x - w // 2:x + w // 2]
    return cropped

# 后面都是没用到的, 是ACS自带的代码。
def categorical_to_one_hot(x, dim=1, expand_dim=False, n_classes=None):
    '''Sequence and label.
    when dim = -1:
    b x 1 => b x n_classes
    when dim = 1:
    b x 1 x h x w => b x n_classes x h x w'''
    # assert (x - x.long().to(x.dtype)).max().item() < 1e-6
    if type(x)==np.ndarray:
        x = torch.Tensor(x)
    assert torch.allclose(x, x.long().to(x.dtype))
    x = x.long()
    if n_classes is None:
        n_classes = int(torch.max(x)) + 1
    if expand_dim:
        x = x.unsqueeze(dim)
    else:
        assert x.shape[dim] == 1
    shape = list(x.shape)
    shape[dim] = n_classes
    x_one_hot = torch.zeros(shape).to(x.device).scatter_(dim=dim, index=x, value=1.)
    return x_one_hot.long()  

def cal_dice_per_case(pred_logit, target, smooth = 1e-8): # target is one hot
    pred_classes = pred_logit.max(dim=1)[1]
    batch_size, n_classes = pred_logit.shape[:2]
    pred_one_hot = categorical_to_one_hot(pred_classes, dim=1, expand_dim=True, n_classes=n_classes)
    intersection = (pred_one_hot * target).view(batch_size, n_classes, -1).sum(-1).float()
    dice = (2 * intersection / (pred_one_hot.view(batch_size, n_classes, -1).\
        sum(-1).float() +\
        target.view(batch_size, n_classes, -1).sum(-1).float() + smooth)).mean(0)
    return dice

def cal_batch_dice(pred_logit, target, smooth = 1e-8): # target is one hot
    pred_classes = pred_logit.max(dim=1)[1]
    batch_size, n_classes = pred_logit.shape[:2]
    pred_one_hot = categorical_to_one_hot(pred_classes, dim=1, expand_dim=True, n_classes=n_classes)
    intersection = (pred_one_hot * target).view(batch_size, n_classes, -1).sum(-1).sum(0).float()
    dice = 2 * intersection / (pred_one_hot.view(batch_size, n_classes, -1).\
        sum(-1).sum(0).float() +\
        target.view(batch_size, n_classes, -1).sum(-1).sum(0).float() + smooth)
    return dice

def cal_iou_per_case(pred_logit, target, smooth = 1e-8): # target is one hot
    pred_classes = pred_logit.max(dim=1)[1]
    batch_size, n_classes = pred_logit.shape[:2]
    pred_one_hot = categorical_to_one_hot(pred_classes, dim=1, expand_dim=True, n_classes=n_classes)
    intersection = (pred_one_hot * target).view(batch_size, n_classes, -1).sum(-1).float()
    iou = (intersection / (pred_one_hot.view(batch_size, n_classes, -1).sum(-1).float() + \
                                target.view(batch_size, n_classes, -1).sum(-1).float() - intersection + smooth)).mean(0)
    return iou

def cal_batch_iou(pred_logit, target, smooth = 1e-8): # target is one hot
    pred_classes = pred_logit.max(dim=1)[1]
    batch_size, n_classes = pred_logit.shape[:2]
    pred_one_hot = categorical_to_one_hot(pred_classes, dim=1, expand_dim=True, n_classes=n_classes)
    intersection = (pred_one_hot * target).view(batch_size, n_classes, -1).sum(-1).sum(0).float()
    iou = intersection / ((pred_one_hot.view(batch_size, n_classes, -1).sum(-1).sum(0).float() + \
                          target.view(batch_size, n_classes, -1).sum(-1).sum(0).float() - intersection + smooth))
    return iou
