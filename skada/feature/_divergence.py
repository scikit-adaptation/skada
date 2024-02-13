# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#
# License: BSD 3-Clause
import torch
from skada.feature.base import (
    DomainAwareModule,
    DomainAwareCriterion,
    DomainBalancedDataLoader,
    DomainAwareNet,
    BaseDALoss,
)
from . import deepcoral_loss


class DeepCoralLoss(BaseDALoss):
    def __init__(self, reg=1):
        super(DeepCoralLoss, self).__init__()
        self.reg = reg

    def forward(
        self,
        y_s,
        y_pred_s,
        y_pred_t,
        domain_pred_s,
        domain_pred_t,
        features_s,
        features_t,
    ):
        """Compute the domain adaptation loss"""
        cov_s = torch.cov(features_s)
        cov_t = torch.cov(features_t)
        loss = self.reg * deepcoral_loss(cov_s, cov_t)
        return loss


def DeepCoral(module, layer_name, reg=1, **kwargs):
    net = DomainAwareNet(
        DomainAwareModule(module, layer_name),
        iterator_train=DomainBalancedDataLoader,
        criterion=DomainAwareCriterion(
            torch.nn.CrossEntropyLoss(), DeepCoralLoss(reg=reg)
        ),
        **kwargs
    )
    return net
