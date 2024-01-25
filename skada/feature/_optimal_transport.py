# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#
# License: BSD 3-Clause
from torch import nn

from skada.feature.base import (
    DomainAwareCriterion,
    DomainBalancedDataLoader,
    DomainAwareNet,
    BaseDALoss,
)

from . import deepjdot_loss


class DeepJDOTLoss(BaseDALoss):
    """Loss DeepJDOT.

    See [1]_.

    Parameters
    ----------
    reg_d : float, default=1
        Distance term regularization parameter.
    reg_cl : float, default=1
        Class distance term regularization parameter.
    target_criterion : torch criterion (class)
        The uninitialized criterion (loss) used to compute the
        DeepJDOT loss. The criterion should support reduction='none'.

    References
    ----------
    .. [1]  Bharath Bhushan Damodaran, Benjamin Kellenberger,
            Remi Flamary, Devis Tuia, and Nicolas Courty.
            DeepJDOT: Deep Joint Distribution Optimal Transport
            for Unsupervised Domain Adaptation. In ECCV 2018
            15th European Conference on Computer Vision,
            September 2018. Springer.
    """
    def __init__(self, reg_d=1, reg_cl=1, target_criterion=None):
        super(DeepJDOTLoss, self).__init__()
        self.reg_d = reg_d
        self.reg_cl = reg_cl
        self.criterion_ = target_criterion

    def forward(
        self,
        y_pred_s,
        y_pred_t,
        y_pred_domain_s,
        y_pred_domain_t,
        features_s,
        features_t,
    ):
        """Compute the domain adaptation loss"""
        loss = deepjdot_loss(
            y_pred_s,
            y_pred_t,
            features_s,
            features_t,
            self.reg_d,
            self.reg_cl,
            criterion=self.target_criterion_,
        )
        return loss


def DeepJDOT(
    module,
    layer_name,
    reg_d=1,
    reg_cl=1,
    target_criterion=nn.CrossEntropyLoss,
    **kwargs
):
    net = DomainAwareNet(
        module,
        layer_name,
        iterator_train=DomainBalancedDataLoader,
        criterion=DomainAwareCriterion(
            nn.CrossEntropyLoss(), DeepJDOTLoss(reg_d, reg_cl, target_criterion)
        ),
        **kwargs
    )
    return net
