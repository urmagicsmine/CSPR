import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..builder import LOSSES


@LOSSES.register_module()
class CombineLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(CombineLoss, self).__init__()

    def forward(self,
            cls_score,
            label,
            loss_weight=(1.0, 0.5),
            **kwargs):
        assert len(loss_weight) == 2
        label_onehot = categorical_to_one_hot(label, expand_dim=True)
        loss = combine_loss(cls_score, label_onehot, loss_weight)
        return loss


def soft_dice_loss(logits, targets, smooth=1.0): # targets is one hot
    probs = logits.softmax(dim=1)
    n_classes = logits.shape[1]
    loss = 0
    for i_class in range(n_classes):
        if targets[:,i_class].sum()>0:
            loss += dice_loss_perclass(probs[:,i_class], targets[:,i_class], smooth)
    return loss / n_classes

def dice_loss_perclass(probs, targets, smooth=1.):
    intersection = probs * targets.float()
    # print(intersection.sum().item(), probs.sum().item()+targets.sum().item())
    if 1 - (2. * intersection.sum()+smooth) / (probs.sum()+targets.sum()+smooth)<0:
        print(intersection.sum().item(), probs.sum().item()+targets.sum().item())
    return 1 - (2. * intersection.sum()+smooth) / (probs.sum()+targets.sum()+smooth)


def soft_cross_entropy_loss(pred_logit, target): # target is one hot
    log_pred = F.log_softmax(pred_logit, dim=-1)
    loss = -(log_pred * target).mean()
    return loss

def combine_loss(pred, targets, loss_weight=(1.0, 0.5)):
    #alpha = 1.0
    #beta = 0.5
    alpha, beta = loss_weight
    # TODO: diceloss should has a weight of 0.5 !!!!
    loss = alpha * soft_cross_entropy_loss(pred, targets) + \
            beta * soft_dice_loss(pred, targets)
    return loss

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
    #print(x.shape, x)
    if expand_dim:
        x = x.unsqueeze(dim)
    else:
        assert x.shape[dim] == 1
    shape = list(x.shape)
    shape[dim] = n_classes
    #print(shape, x.shape)
    x_one_hot = torch.zeros(shape).to(x.device).scatter_(dim=dim, index=x, value=1.)
    return x_one_hot.long()


