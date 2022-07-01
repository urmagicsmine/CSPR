#from typing import ForwardRef
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.autograd import Function, Variable
from ..builder import LOSSES
from mmcv import Timer

from .cross_entropy_loss import *

print_tensor = lambda n, x: print(n, type(x), x.shape, x.min(), x.max())

def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None,
                         ignore_index = None):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.

    Returns:
        torch.Tensor: The calculated loss
    """

    if pred.dim() != label.dim():
        if pred.size(1) == 1:
            label = label.unsqueeze(1)
        else:
            label = One_Hot(pred.size(1))(label)#_expand_onehot_labels(label, weight, pred.size(-1))
            
    if ignore_index is not None: label[label == ignore_index] = 0
    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), weight=class_weight, reduction='none')
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss

@LOSSES.register_module()
class ComboLoss(nn.Module):
    """ComboLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float], optional): Weight of each class.
            Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=(1.0, 0.5),
                 num_classes = 1,
                 uncertain_map_alpha = None,
                 verbose = False,
                 dice_alpha = 1, 
                 dice_beta = 1,
                 smooth=1.
                 ):
        super(ComboLoss, self).__init__()

        assert len(loss_weight) == 2
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weights = loss_weight
        self.class_weight = class_weight
        self.uncertain_map_alpha = uncertain_map_alpha
        self.dice_alpha = dice_alpha
        self.dice_beta = dice_beta
        self.verbose = verbose
        # self.pos_topk = pos_topk
        self.nb_classes = num_classes
        self.smooth = smooth
        if self.loss_weights[0] != 0:
            self.criterion_1 = cross_entropy if num_classes > 1 else binary_cross_entropy
        else:
            self.criterion_1  = None
        
        if self.loss_weights[1] != 0:
            self.criterion_2 = SoftDiceLoss(num_classes, uncertain_map_alpha = uncertain_map_alpha,
                                            alpha = self.dice_alpha, beta = self.dice_beta, smooth=self.smooth) 
        else:
            self.criterion_2 = None

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        # the diceloss will do inplace operation on gt. In cascade scenario, GT is needed in multiple decoder heads to compute gradient
        # where inplace operation is not allowed. this issue can be solved by cloning GT every time the loss fucntion is called.
        label = label.clone().detach()
        if self.verbose: print_tensor('\n %0.2f pred' %sum(self.loss_weights), cls_score)
        if self.verbose: print_tensor('true', label)

        loss_1 = self.loss_weights[0] * self.criterion_1(
            cls_score,
            label,
            weight,
            class_weight=self.class_weight,
            reduction=self.reduction,
            avg_factor=avg_factor,
            **kwargs) if self.loss_weights[0] != 0 else 0

        loss_2 = self.loss_weights[1] * self.criterion_2(
                    cls_score, 
                    label) if self.loss_weights[1] != 0 else 0

        if self.verbose: print_tensor('BCE', loss_1)
        if self.verbose: print_tensor('Dloss', loss_2)
        total_loss = loss_1 + loss_2 
        return total_loss

class SoftDiceLoss(nn.Module):
    """
    a generalized class of dice loss where
    the ratio of false positive and false negative can be specified in the formula
    alpha stands for the contribution of false positive to the metric
    beta stands for the contribution of false negative to the metric

    To increase recall, one can increase beta to penalize false negative
    To increase precision, one can increase alpha

    """

    def __init__(self, n_classes, uncertain_map_alpha = 0, alpha = 1, beta = 1, ignore_0 = True, smooth=1.):
        super(SoftDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes) #.forward
        self.n_classes = n_classes
        self.uncertain_map_alpha = uncertain_map_alpha
        self.alpha = alpha
        self.beta = beta 
        self.ignore_0 = ignore_0
        self.smooth = smooth

    def forward(self, pred, gt):
        #self.smooth = 1e-3
        batch_size = pred.size(0)     
        pred_temp = torch.softmax(pred, dim=1) if self.n_classes > 1 else torch.sigmoid(pred)
        pred_temp = pred_temp.view(batch_size, self.n_classes, -1)

        gt_temp = self.one_hot_encoder(gt)
        gt_temp = gt_temp.contiguous().view(batch_size, self.n_classes, -1) #

        if self.alpha != 1 or self.beta != 1:
            tp = torch.sum(pred_temp * gt_temp, 2) #+ self.smooth
            fp = torch.sum(pred_temp * (1.0 - gt_temp), 2)
            fn = torch.sum((1.0- pred_temp) * gt_temp, 2)
            dice_by_sample_class = (2.0 * tp + self.smooth) / ( self.alpha * fp + 2.0 * tp + self.beta * fn + self.smooth)
        else:
            intersection = torch.sum(pred_temp * gt_temp, 2) #+ self.smooth
            union = torch.sum(pred_temp + gt_temp, 2)
            dice_by_sample_class = (2.0 * intersection + self.smooth * (union ==0)) / ( intersection + union + self.smooth)  #* (union == 0)
        mdice_by_sample = torch.sum(dice_by_sample_class, 0)/float(batch_size)
        start_class = 1 if (self.ignore_0 and self.n_classes > 1) else 0
        mdice_by_class = torch.sum(mdice_by_sample[start_class:])/float(self.n_classes - start_class)
        score = 1.0 - mdice_by_class
        return score

def get_uncertain_maps(pred_tensor, true_tensor, threshold = 0.37):
    """
    
    an implementation of the uncertain map proposed by 
        Zheng H, Chen Y, Yue X, et al. 
        Deep pancreas segmentation with uncertain regions of shadowed sets[J]. 
        Magnetic Resonance Imaging, 2020: 45-52.

    separation threshold: α split the prediction as three sets, certain foreground, certain background and uncertain region
    Prediction map: P
    Grounth truth map: Y
    uncertain weight map C 

    Ci = | Pi - Yi | (if α < Pi < 1- α) else 1

    args: 
        pred_tensor: prediciton probability in range(0, 1); post sigmoid or softmax!
        true_tensor: true label in range(0, 1)

    """
    certain_mask = (pred_tensor < threshold) & (pred_tensor > (1 - threshold))
    uncertain_mask = ~certain_mask
    uncertain_map = torch.abs(pred_tensor - true_tensor) * uncertain_mask + certain_mask
    return uncertain_map.long()


class CustomSoftDiceLoss(nn.Module):
    def __init__(self, n_classes, class_ids, smooth=1.):
        super(CustomSoftDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes).forward
        self.n_classes = n_classes
        self.class_ids = class_ids
        self.smooth = smooth

    def forward(self, input, target):
        #self.smooth = 0.01
        batch_size = input.size(0)

        input = F.softmax(input[:,self.class_ids], dim=1).view(batch_size, len(self.class_ids), -1)
        target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)
        target = target[:, self.class_ids, :]


        inter = torch.sum(input * target, 2) + self.smooth
        union = torch.sum(input, 2) + torch.sum(target, 2) + self.smooth

        score = torch.sum(2.0 * inter / union)
        score = 1.0 - score / (float(batch_size) * float(self.n_classes))

        return score


class One_Hot(object):
    """transform the value in mask into one-hot representation
        depth: number of unique value in the mask
    """
    def __init__(self, depth):
        # super(One_Hot, self).__init__()
        self.depth = depth
        self.ones = torch.sparse.torch.eye(depth).cuda() # identity matrix, diagnal is 1 

    def __call__(self, X_in : Tensor):
        if self.depth <= 1:
            return X_in.unsqueeze(1)

        n_dim = X_in.dim()
        output_size = X_in.size() + torch.Size([self.depth])
        num_element = X_in.numel()
        X_in = X_in.data.long().view(num_element) # flatten X_in into a long vector
        out = Variable(self.ones.index_select(0, X_in)).view(output_size) # using label value as indexer to create one-hot
        return out.permute(0, -1, *range(1, n_dim)).squeeze(dim=2).float()

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)



def cross_entropy_2D(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(target.numel())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= float(target.numel())
    return loss


def cross_entropy_3D(input, target, weight=None, size_average=True):
    n, c, h, w, s = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous().view(-1, c)
    target = target.view(target.numel())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= float(target.numel())
    return loss


if __name__ == '__main__':
    from torch.autograd import Variable
    depth=3
    batch_size=2
    encoder = One_Hot(depth=depth).forward
    y = Variable(torch.LongTensor(batch_size, 1, 1, 2 ,2).random_() % depth).cuda()  # 4 classes,1x3x3 img
    y_onehot = encoder(y)
    x = Variable(torch.randn(y_onehot.size()).float()).cuda()
    dicemetric = SoftDiceLoss(n_classes=depth)
    dicemetric(x,y)
