from .accuracy import Accuracy
from .auc import Auc
from ..builder import HEADS, build_loss
from .base_head import BaseHead
import pdb

@HEADS.register_module()
class ClsHead(BaseHead):
    """classification head.

    Args:
        loss (dict): Config of classification loss.
        topk (int | tuple): Top-k accuracy.
    """  # noqa: W605

    def __init__(self,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk=(1, )):
        super(ClsHead, self).__init__()

        assert isinstance(loss, dict)
        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk, )
        for _topk in topk:
            assert _topk > 0, 'Top-k should be larger than 0'
        self.topk = topk

        self.compute_loss = build_loss(loss)
        self.compute_accuracy = Accuracy(topk=self.topk)
        self.compute_auc = Auc()

    # LJ
    def loss(self, cls_score, gt_label, multi_cls=False):
        num_samples = len(cls_score)
        losses = dict()
        # compute loss
        loss = self.compute_loss(cls_score, gt_label, avg_factor=num_samples)
        # compute accuracy
        if multi_cls:
            auc_mean = self.compute_auc(cls_score, gt_label)
            losses['auc_mean'] = auc_mean
        else:
            acc = self.compute_accuracy(cls_score, gt_label)
            losses['accuracy'] = {f'top-{k}': a for k, a in zip(self.topk, acc)}

        losses['loss'] = loss
        # pdb.set_trace()

        return losses

    def forward_train(self, cls_score, gt_label):
        losses = self.loss(cls_score, gt_label)
        return losses