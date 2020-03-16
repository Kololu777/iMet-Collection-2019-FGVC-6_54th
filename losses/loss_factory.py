import torch.nn as nn
from .Focalloss import FocalLoss
from .lovaszloss import lovasz_hinge
from .FbetaLoss import FbetaLoss


class CombineLoss(nn.Module):
    def __init__(self):
        super(CombineLoss, self).__init__()
        self.fbeta_loss = FbetaLoss(beta=2)
        self.focal_loss = FocalLoss()
        # self.lovasz_loss=lovasz_hinge()

    def forward(self, logits, labels):

        loss_focal = self.focal_loss(logits, labels)
        lovasz = lovasz_hinge(logits, labels)
        return 0.5 * lovasz + 0.5 * loss_focal


class CombineBetaLoss(nn.Module):
    def __init__(self):
        super(CombineLoss, self).__init__()
        self.fbeta_loss = FbetaLoss(beta=2)
        self.focal_loss = FocalLoss()
        # self.lovasz_loss=lovasz_hinge()

    def forward(self, logits, labels):
        loss_beta = self.fbeta_loss(logits, labels)
        loss_focal = self.focal_loss(logits, labels)
        return 0.5 * loss_beta + 0.5 * loss_focal


def select_criterion(criterion_name):
    if criterion_name == 'bce':
        return nn.BCEWithLogitsLoss()
    elif criterion_name == 'focal':
        return FocalLoss()
    elif criterion_name == 'comb':
        return CombineLoss()
    elif criterion_name == 'comb_2':
        return CombineBetaLoss()
