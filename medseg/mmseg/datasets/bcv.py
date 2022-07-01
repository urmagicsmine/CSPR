from torch.utils.data import Dataset
import random
import os
import os.path as osp
import numpy as np
import pandas as pd
import torch

import mmcv
from mmcv.utils import print_log
from mmseg.core import mean_dice

from .custom import CustomDataset
from .builder import DATASETS

from .pipelines.load_utils import load_image_nii, array2nii, get_affmat, save_nii, itk_load_nii

from sklearn.model_selection import train_test_split

@DATASETS.register_module()
class BCVDataset(CustomDataset):
    # 13-classes
    CLASSES = (
         "background",
         "spleen",
         "kidney_right",
         "kidney_left",
         "gallbladder",
         "esophagus",
         "liver",
         "stomach",
         "aorta",
         "ivc",
         "pvsv",
         "pancreas",
         "adrenal_gland_right",
         "adrenal_gland_left"
         )
    PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
               [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
               [150, 5, 61]]

    def __init__(self, ann_file, random_split=False, img_dir='images', size=None, crop_size=(32,256,256), disp_per_case=False, *args, **kwargs):
        self.random_split = random_split
        self.img_dir=img_dir
        self.size = size
        self.crop_size = crop_size
        self.ann_file = ann_file
        self.disp_per_case = False
        super(BCVDataset, self).__init__(img_dir=img_dir, *args, **kwargs)

    def load_annotations(self, *args, **kwargs):
        with open(os.path.join(self.data_root, self.ann_file), 'r') as f:
            lines = f.readlines()
        #names = [os.path.join(self.data_root, self.img_dir, line.strip()) for line in lines]
        names = [os.path.join(self.img_dir, line.strip()) for line in lines]
        if self.random_split:
            train_names, test_names = train_test_split(names, test_size=0.3, random_state=0)
        else:
            # as same as UNETR
            train_names = names[:24]
            test_names = names[24:]

        self.names = test_names if self.test_mode else train_names
        #self.names = train_names
        #print('train_names,', train_names)
        #print('test_names,', test_names)
        print('Load img_infos is ', self.names)
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
            # used for lidc-style data transfrom
            if not self.size is None:
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
        # calculate all acc, per case acc, all dice, per case dice.
        all_acc, acc, dice, case_dice = mean_dice(
            results, gt_seg_maps, num_classes, ignore_index=self.ignore_index)
        # Show dice for each case
        #for i in range(len(results)):
            #print('idx {}, pred_sum {},\tgt_sum {},\tdice {:.2f}, \t {}'\
                    #.format(i, results[i].sum(), gt_seg_maps[i].sum(), case_dice[i][1],\
                    #self.img_infos[i]['filename']))
        summary_str = '\n'
        if self.disp_per_case:
            for ind in range(len(results)):
                summary_str += '%-25s:' % osp.basename(self.img_infos[ind]['filename']) + str(case_dice[ind]) + '\n'
            summary_str += "MeanDice per case: " + str(np.mean(case_dice, axis=0)) + '\n'
        '''
        # case_dice: NxM list, where N is num of test imgs, M is num of Classes.
        #print(type(case_dice), case_dice)
        #print(type(case_dice[0]))
        print(gt_seg_maps[0].shape)
        gt_all = np.stack(gt_seg_maps, axis=0)
        gt_all = torch.Tensor(gt_all)
        pred_all = np.stack(results, axis=0)
        pred_all = torch.Tensor(pred_all)
        print(gt_all.shape)
        gt_one_hot = categorical_to_one_hot(gt_all, expand_dim=True)
        dice_total = cal_batch_dice(pred_all, gt_one_hot)
        dice_per_case = cal_dice_per_case(pred_all, gt_one_hot)
        #print('ACS dice per_case {:.2f}, global {:.2f}'.format(dice_per_case, dice_total))
        print('ACS dice global , per_case ', dice_per_case, dice_total, type(dice_per_case), type(dice_total))
        '''

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

        # for debug
        #print('\nall_acc', all_acc,'\nacc', acc, '\ndice', dice, '\ncase_dice', case_dice)
        #print('len of acc', len(acc), len(class_names))
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

    def format_results(self, results, imgfile_prefix=None, to_label_id=False):
        """Format the results into dir (standard format for LTIS evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str | None): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
                the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """

        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        if imgfile_prefix is None:
            #tmp_dir = tempfile.TemporaryDirectory()
            #imgfile_prefix = tmp_dir.name
            imgfile_prefix = './work_dirs/bcv_visual'
            tmp_dir='./work_dirs/bcv_visual'
        else:
            tmp_dir = None
        print('\nsaveing nii to:', imgfile_prefix)
        result_files = self.results2img(results, imgfile_prefix, to_label_id)

        return result_files, tmp_dir

    def results2img(self, results, imgfile_prefix, to_label_id):
        """Write the segmentation results to images.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        prog_bar = mmcv.ProgressBar(len(self))
        for idx in range(len(self)):
            result = results[idx]
            if to_label_id:
                result = self._convert_to_label_id(result)
            filename = self.img_infos[idx]['filename']
            with np.load(filename) as npz:
                img, gt_seg_map = npz['voxel'], npz['answer1']
            shape = gt_seg_map.shape
            center = np.array(shape) // 2
            #gt_seg_map = self.center_crop(center, gt_seg_map)
            #img = self.center_crop(center, img)
            #affine_matrix = get_affmat(filename)
            basename = osp.splitext(osp.basename(filename))[0]

            pred_nii_filename = osp.join(imgfile_prefix, '%s_pred.nii' % basename)
            gt_nii_filename = osp.join(imgfile_prefix, '%s_gt.nii' % basename)
            img_nii_filename = osp.join(imgfile_prefix, '%s_img.nii' % basename)

            # here img and gt_seg_map is hwd style, result is dhw style
            # we need to transpose them into dhw style so as to save as nii file.
            #result = result.transpose(1, 2, 0) # to hwd
            img = img.transpose(2, 0, 1) # hwd to dhw
            gt_seg_map = gt_seg_map.transpose(2, 0, 1) # hwd to dhw
            print(img.shape, result.shape, gt_seg_map.shape)
            pred_nii_fp = save_nii(result.astype('uint8'), pred_nii_filename, image=None)
            gt_nii_fp = save_nii(gt_seg_map.astype('uint8'), gt_nii_filename, image=None)
            img_nii_fp = save_nii(img, img_nii_filename, image=None)

            result_files.append(pred_nii_fp)
            prog_bar.update()

        return result_files

    def center_crop(self, crop_center, img):
        y, x, z = crop_center
        h, w, d = self.crop_size
        cropped = img[y - h // 2:y + h // 2,
                      x - w // 2:x + w // 2,
                      z - d // 2:z + d // 2]
        return cropped

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

def cal_dice_per_case(pred, target, smooth = 1e-8): # target is one hot
    batch_size = pred.shape[0]
    n_classes = 2
    pred_one_hot = categorical_to_one_hot(pred, dim=1, expand_dim=True)
    intersection = (pred_one_hot * target).view(batch_size, n_classes, -1).sum(-1).float()
    dice = (2 * intersection / (pred_one_hot.view(batch_size, n_classes, -1).\
        sum(-1).float() +\
        target.view(batch_size, n_classes, -1).sum(-1).float() + smooth)).mean(0)
    return dice

def cal_batch_dice(pred, target, smooth = 1e-8): # target is one hot
    batch_size = pred.shape[0]
    n_classes = 2
    pred_one_hot = categorical_to_one_hot(pred, dim=1, expand_dim=True)
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
