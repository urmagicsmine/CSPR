from .custom import CustomDataset
from .pipelines.image_io_new import load_multislice_gray_png_new
from .builder import DATASETS
import os.path as osp
import numpy as np
from mmseg.utils import get_root_logger
from mmcv.utils import print_log
import pdb
import sys
from mmseg.core import mean_dice
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from ..models.classifier_head.accuracy import accuracy


@DATASETS.register_module()
class PneumoniaDataset(CustomDataset):

    # CLASSES = ('background', 'consolidation', 'ground_glass_opacity')
    CLASSES = ('background', 'lesion')

    def __init__(self, num_slice=32, wclassifier=False, disp_per_case=False,  **kwargs):
        super(PneumoniaDataset, self).__init__(
            img_dir=None,
            reduce_zero_label=False,
            img_suffix=None,
            seg_map_suffix=None,
            **kwargs)
        self.num_slice = num_slice
        self.disp_per_case = disp_per_case
        self.wclassifier = wclassifier


    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        '''Overwrite load_annotations func.
        '''

        img_infos = []
        if split is not None:
            with open(split, 'r') as f:
                lines = f.readlines()
            for line in lines:
                if len(line.strip().split()) == 5:
                    # Added to support 'image_path seg_path'
                    img_name, seg_name, min_idx, max_idx, label = line.strip().split()
                    png_range = (int(min_idx), int(max_idx))
                    _label = 1 if label == 'pos' else 0
                    # 只保留正样本
                    # if _label == 0: continue

                    img_info = dict(filename=osp.join(self.data_root, img_name))
                    img_info['ann'] = dict(seg_map=osp.join(self.data_root, seg_name))
                    img_info['png_range'] = dict(png_range=png_range)
                    # 训练和测试处理逻辑不同 训练时切patch后更新gt_label
                    img_info['gt_label'] = np.array(_label, dtype=np.int64)
                    img_infos.append(img_info)
                else:
                    raise NotImplementedError
        else:
            raise NotImplementedError
        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos


    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        range_info = self.img_infos[idx]['png_range']
        if self.wclassifier:
            gt_label = self.img_infos[idx]['gt_label']
            results = dict(img_info=img_info, ann_info=ann_info, range_info=range_info, gt_label=gt_label)
        else:
            results = dict(img_info=img_info, ann_info=ann_info, range_info=range_info)
        self.pre_pipeline(results)
        return self.pipeline(results)


    def prepare_test_img(self, idx):

        img_info = self.img_infos[idx]
        range_info = self.img_infos[idx]['png_range']
        results = dict(img_info=img_info, range_info=range_info)
        self.pre_pipeline(results)
        return self.pipeline(results)


    def evaluate(self, results, metric='auc_and_mDice3D', logger=None, post_process=False, **kwargs):
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
        allowed_metrics = ['auc_and_mDice3D', 'mDice3D']
        if metric not in allowed_metrics:
            raise KeyError('metric {} is not supported'.format(metric))
        eval_results = {}

        # 分类分支
        if isinstance(results[0], dict):
            metric_options = {'topk': (1, 2)}
            seg_results = []
            classifier_results = []
            for idx in range(len(results)):
                seg_results += results[idx]['seg_pred']
                classifier_results += results[idx]['classifier_pred']

            # 分类
            topk = metric_options.get('topk')
            classifier_results = np.vstack(classifier_results)
            gt_labels = self.get_gt_labels()
            num_imgs = len(classifier_results)
            assert len(gt_labels) == num_imgs
            acc = accuracy(classifier_results, gt_labels, topk)
            # loss = F.nll_loss(torch.log(torch.Tensor(classifier_results)), torch.Tensor(gt_labels).long()).mean()
            # loss = round(float(loss), 5)
            # 只保留正阳本时 去掉auc计算
            auc = roc_auc_score(gt_labels, classifier_results[:, 1])
            eval_results = {f'top-{k}': a.item() for k, a in zip(topk, acc)}
            # eval_results['loss'] = loss
            eval_results['auc'] = auc

            print('top1-acc:%.4f' % acc[0])
            print('auc:%.4f' % auc)
        else:
            seg_results = results

        # 分割
        gt_seg_maps = self.get_gt_seg_maps()

        gt_seg_maps = [gt_seg_map.transpose(2, 0, 1) for gt_seg_map in gt_seg_maps]
        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)
        # 验证时无post_process 测试时有post_process 导致结果不一样 现已统一
        all_acc, acc, dice, case_dice = mean_dice(
            seg_results, gt_seg_maps, num_classes, ignore_index=self.ignore_index, post_process=post_process)
        summary_str = '\n'
        if self.disp_per_case:
            for ind in range(len(seg_results)):
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


    def get_gt_seg_maps(self):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        for img_info in self.img_infos:
            seg_map = img_info['ann']['seg_map']
            min_idx = img_info['png_range']['png_range'][0]
            max_idx = img_info['png_range']['png_range'][1]

            # 取整块肺的区域
            sampled_idx = (min_idx + max_idx) // 2
            seg_map = osp.join(seg_map, '%03d.png' % sampled_idx)
            # 若肺区域跨度很小 则不更新num_slice
            if (max_idx - min_idx + 1) > self.num_slice:
                num_slice_update = max_idx - min_idx + 1
            else:
                num_slice_update = self.num_slice
            gt_seg_map = load_multislice_gray_png_new(seg_map, min_idx, max_idx, num_slice_update)
            # 合并类别
            gt_seg_map[gt_seg_map != 0] = 1

            # modify if custom classes
            if self.label_map is not None:
                for old_id, new_id in self.label_map.items():
                    gt_seg_map[gt_seg_map == old_id] = new_id
            if self.reduce_zero_label:
                # avoid using underflow conversion
                gt_seg_map[gt_seg_map == 0] = 255
                gt_seg_map = gt_seg_map - 1
                gt_seg_map[gt_seg_map == 254] = 255

            gt_seg_maps.append(gt_seg_map)

        return gt_seg_maps


    def get_gt_labels(self):
        """Get all ground-truth labels (categories).

        Returns:
            list[int]: categories for all images.
        """
        gt_labels = np.array([data['gt_label'] for data in self.img_infos])

        return gt_labels