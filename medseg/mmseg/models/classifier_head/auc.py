import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import hamming_loss, accuracy_score, roc_auc_score, roc_curve, auc

def auc_multi_cls(pred, target):
    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        device_id = torch.cuda.current_device()
        pred = torch.sigmoid(pred)
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()

        auc_total = 0.0
        cls_num = pred.shape[1]
        for idx in range(cls_num):
            try:
                _auc = roc_auc_score(target[:, idx], pred[:, idx])
            # ValueError: Only one class present in y_true.ROC AUC score is not defined in that case.
            # target[:, idx] = array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            except ValueError:
                _auc = 0.5
                # _auc = 0.0
                # cls_num = cls_num - 1

            auc_total += _auc

        auc_mean = auc_total / cls_num
        auc_mean = torch.tensor(auc_mean).cuda(device_id)

    return auc_mean


# LJ
class Auc(nn.Module):

    def __init__(self):
        """Module to calculate the auc

        """
        super().__init__()


    def forward(self, pred, target):
        """Forward function to calculate accuracy

        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.

        Returns:
            tuple[float]: The accuracies under different topk criterions.
        """
        # LJ
        return auc_multi_cls(pred, target)
