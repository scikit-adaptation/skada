# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#
# License: BSD 3-Clause
from torch import nn

from skada.deep.base import (
    BaseDALoss,
    DomainAwareCriterion,
    DomainAwareModule,
    DomainAwareNet,
    DomainBalancedDataLoader,
)

from .losses import deepjdot_loss


class DeepJDOTLoss(BaseDALoss):
    """Loss DeepJDOT.

    This loss reduces the distance between source and target domain
    through a measure of discrepancy on joint deep
    representations/labels based on optimal transport.
    See [13]_.

    Parameters
    ----------
    reg_cl : float, default=1
        Class distance term regularization parameter.
    target_criterion : torch criterion (class)
        The uninitialized criterion (loss) used to compute the
        DeepJDOT loss. The criterion should support reduction='none'.

    References
    ----------
    .. [13]  Bharath Bhushan Damodaran, Benjamin Kellenberger,
            Remi Flamary, Devis Tuia, and Nicolas Courty.
            DeepJDOT: Deep Joint Distribution Optimal Transport
            for Unsupervised Domain Adaptation. In ECCV 2018
            15th European Conference on Computer Vision,
            September 2018. Springer.
    """

    def __init__(self, reg_cl=1, target_criterion=None):
        super().__init__()
        self.reg_cl = reg_cl
        self.criterion_ = target_criterion

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
        loss = deepjdot_loss(
            y_s,
            y_pred_t,
            features_s,
            features_t,
            self.reg_cl,
            criterion=self.criterion_,
        )
        return loss


def DeepJDOT(module, layer_name, reg=1, reg_cl=1, target_criterion=None, **kwargs):
    """DeepJDOT.

       See [13]_.

    Parameters
    ----------
    module : torch module (class or instance)
        A PyTorch :class:`~torch.nn.Module`.
    layer_name : str
        The name of the module's layer whose outputs are
        collected during the training for the adaptation.
    reg : float, default=1
        Regularization parameter.
    reg_cl : float, default=1
        Class distance term regularization parameter.
    target_criterion : torch criterion (class)
        The uninitialized criterion (loss) used to compute the
        DeepJDOT loss. The criterion should support reduction='none'.

    References
    ----------
    .. [13]  Bharath Bhushan Damodaran, Benjamin Kellenberger,
            Remi Flamary, Devis Tuia, and Nicolas Courty.
            DeepJDOT: Deep Joint Distribution Optimal Transport
            for Unsupervised Domain Adaptation. In ECCV 2018
            15th European Conference on Computer Vision,
            September 2018. Springer.
    """
    net = DomainAwareNet(
        module=DomainAwareModule,
        module__base_module=module,
        module__layer_name=layer_name,
        iterator_train=DomainBalancedDataLoader,
        criterion=DomainAwareCriterion,
        criterion__criterion=nn.CrossEntropyLoss(),
        criterion__adapt_criterion=DeepJDOTLoss(reg_cl, target_criterion),
        criterion__reg=reg,
        **kwargs,
    )
    return net
