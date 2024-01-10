# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#
# License: BSD 3-Clause
import torch
from skorch import NeuralNetClassifier
from skada.feature.base import (
    DomainAwareModule,
    DomainAwareCriterion,
    DomainBalancedDataLoader,
)
from . import deepcoral_loss


class DeepCoralLoss(torch.nn.Module):
    def __init__(self, reg=1):
        super(DeepCoralLoss, self).__init__()
        self.reg = reg

    def forward(self, yt, features_s, features_t):
        """Compute the domain adaptation loss"""
        cov_s = torch.cov(features_s)
        cov_t = torch.cov(features_t)
        loss = self.reg * deepcoral_loss(cov_s, cov_t)
        return loss


def DeepCoral(module, reg=1):
    net = NeuralNetClassifier(
        DomainAwareModule(module, "dropout"),
        max_epochs=10,
        lr=0.1,
        iterator_train=DomainBalancedDataLoader,
        iterator_train__batch_size=8,
        train_split=None,
        criterion=DomainAwareCriterion(
            torch.nn.CrossEntropyLoss(), DeepCoralLoss(reg=reg)
        ),
    )
    return net
