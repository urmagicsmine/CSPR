import copy
from abc import ABCMeta, abstractmethod

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset

from mmcls.models.losses import accuracy, cross_entropy
from mmcls.models.losses import get_sensitivity, get_specificity, get_precision, get_F1, get_accuracy
from .pipelines import Compose

from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

import pdb

class BaseDataset(Dataset, metaclass=ABCMeta):
    """Base dataset.

    Args:
        data_prefix (str): the prefix of data path
        pipeline (list): a list of dict, where each element represents
            a operation defined in `mmcls.datasets.pipelines`
        ann_file (str | None): the annotation file. When ann_file is str,
            the subclass is expected to read from the ann_file. When ann_file
            is None, the subclass is expected to read according to data_prefix
        test_mode (bool): in train mode or test mode
    """

    CLASSES = None

    def __init__(self, data_prefix, pipeline, ann_file=None, sub_set=None, test_mode=False):
        super().__init__()

        self.ann_file = ann_file
        self.sub_set = sub_set
        self.data_prefix = data_prefix
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)
        self.data_infos = self.load_annotations()

    @abstractmethod
    def load_annotations(self):
        pass

    @property
    def class_to_idx(self):
        """Map mapping class name to class index.

        Returns:
            dict: mapping from class name to class index.
        """

        return {_class: i for i, _class in enumerate(self.CLASSES)}

    def get_gt_labels(self):
        """Get all ground-truth labels (categories).

        Returns:
            list[int]: categories for all images.
        """

        gt_labels = np.array([data['gt_label'] for data in self.data_infos])
        return gt_labels

    def get_cat_ids(self, idx):
        """Get category id by index.

        Args:
            idx (int): Index of data.

        Returns:
            int: Image category of specified index.
        """

        return self.data_infos[idx]['gt_label'].astype(np.int)

    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        return self.prepare_data(idx)

    def evaluate(self,
                 results,
                 metric='accuracy',
                 metric_options={'topk': (1, 5)},
                 logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is `accuracy`.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
        Returns:
            dict: evaluation results
        """
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        #allowed_metrics = ['accuracy']
        #if metric not in allowed_metrics:
            #raise KeyError(f'metric {metric} is not supported')

        eval_results = {}
        if metric == 'accuracy':
            topk = metric_options.get('topk')
            results = np.vstack(results)
            gt_labels = self.get_gt_labels()
            num_imgs = len(results)
            assert len(gt_labels) == num_imgs
            acc = accuracy(results, gt_labels, topk)
            loss = cross_entropy(torch.Tensor(results), torch.Tensor(gt_labels).long()).mean()
            loss = round(float(loss), 3)
            eval_results = {f'top-{k}': a.item() for k, a in zip(topk, acc)}
            eval_results['loss'] = loss
        elif metric == 'acc_and_auc':
            topk = metric_options.get('topk')
            results = np.vstack(results)
            gt_labels = self.get_gt_labels()
            num_imgs = len(results)
            assert len(gt_labels) == num_imgs
            acc = accuracy(results, gt_labels, topk)
            loss = cross_entropy(torch.Tensor(results), torch.Tensor(gt_labels).long()).mean()
            loss = round(float(loss), 3)
            auc = roc_auc_score(gt_labels, results[:,1])
            eval_results = {f'top-{k}': a.item() for k, a in zip(topk, acc)}
            eval_results['loss'] = loss
            eval_results['auc'] = auc
        elif metric == 'all':
            # gt_labels.shape (n,) n is num of samples
            # results.shape(n, 2) 2 is n_classes
            topk = metric_options.get('topk')
            results = np.vstack(results)
            gt_labels = self.get_gt_labels()
            num_imgs = len(results)
            assert len(gt_labels) == num_imgs
            loss = cross_entropy(torch.Tensor(results), torch.Tensor(gt_labels).long()).mean()
            loss = round(float(loss), 3)
            auc = roc_auc_score(gt_labels, results[:,1])
            #acc = accuracy(results, gt_labels, topk)
            #prec, rec, f1, _ = precision_recall_fscore_support(gt_labels, results[:, 1].round(), average="binary")
            #specificity = get_specificity(torch.Tensor(results[:,1]), torch.Tensor(gt_labels))
            prec, rec, f1, specificity, acc = self.get_best_metrics(gt_labels, results[:, 1])


            #eval_results = {f'top-{k}': a.item() for k, a in zip(topk, acc)}
            eval_results['loss'] = loss
            eval_results['acc'] = acc
            eval_results['auc'] = auc
            eval_results['f1'] = f1
            eval_results['recall'] = rec
            eval_results['precision'] = prec
            eval_results['specificity'] = specificity
        return eval_results
    def get_best_metrics(self, gts, preds):
        thresh_array = np.arange(0,1,0.1)
        res = 0
        best_thresh = 0
        for thresh in thresh_array:
            specificity = get_specificity(torch.Tensor(preds), torch.Tensor(gts), thresh)
            sensitivity = get_sensitivity(torch.Tensor(preds), torch.Tensor(gts), thresh)
            tmp_metric = specificity + sensitivity
            tmp = (specificity * sensitivity) / (specificity + sensitivity)
            if tmp_metric > res:
                res = tmp_metric
                best_thresh = thresh
            #print('thre, spe, sen, sum, f1 {:.2f},{:.2f},{:.2f},{:.2f},{:.2f}'.format(thresh, specificity,sensitivity, tmp_metric, tmp))
        specificity = get_specificity(torch.Tensor(preds), torch.Tensor(gts), best_thresh)
        acc = get_accuracy(torch.Tensor(preds), torch.Tensor(gts), best_thresh)
        preds[preds > best_thresh] = 1.0
        preds[preds <= best_thresh] = 0.0
        prec, rec, f1, _ = precision_recall_fscore_support(gts, preds, average="binary")
        return prec, rec, f1, specificity, acc





