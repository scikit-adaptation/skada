# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#
# License: BSD 3-Clause
import torch

from skada.deep.base import (
    BaseDALoss,
    DomainAwareCriterion,
    DomainAwareModule,
    DomainAwareNet,
    DomainBalancedDataLoader,
)

from .losses import mcc_loss


class MCCLoss(BaseDALoss):
    """Loss MCC.

    This loss reduces the class confusion of the predicted label of target domain
    See [33]_.

    Parameters
    ----------
    T : float, default=1
        Temperature parameter for the scaling.
        If T=1, the scaling is a softmax function.
    eps : float, default=1e-7
        Small constant added to median distance calculation for numerical stability.

    References
    ----------
    .. [33] Ying Jin, Ximei Wang, Mingsheng Long, Jianmin Wang.
            Minimum Class Confusion for Versatile Domain Adaptation.
            In ECCV, 2020.
    """

    def __init__(self, T=1, eps=1e-7):
        super().__init__()
        self.T = T
        self.eps = eps

    def forward(
        self,
        y_pred_t,
        **kwargs,
    ):
        """Compute the domain adaptation loss"""
        loss = mcc_loss(
            y_pred_t,
            T=self.T,
            eps=self.eps,
        )
        return loss


def MCC(
    module,
    layer_name,
    reg=1,
    T=1,
    base_criterion=None,
    **kwargs,
):
    """MCC.

       See [33]_.

    Parameters
    ----------
    module : torch module (class or instance)
        A PyTorch :class:`~torch.nn.Module`.
    layer_name : str
        The name of the module's layer whose outputs are
        collected during the training for the adaptation.
    reg : float, default=1
        Regularization parameter for DA loss.
    T : float, default=1
        Temperature parameter for the scaling.
    base_criterion : torch criterion (class)
        The base criterion used to compute the loss with source
        labels. If None, the default is `torch.nn.CrossEntropyLoss`.

    References
    ----------
    .. [33] Ying Jin, Ximei Wang, Mingsheng Long, Jianmin Wang.
            Minimum Class Confusion for Versatile Domain Adaptation.
            In ECCV, 2020.
    """
    if base_criterion is None:
        base_criterion = torch.nn.CrossEntropyLoss()

    net = DomainAwareNet(
        module=DomainAwareModule,
        module__base_module=module,
        module__layer_name=layer_name,
        iterator_train=DomainBalancedDataLoader,
        criterion=DomainAwareCriterion,
        criterion__base_criterion=base_criterion,
        criterion__adapt_criterion=MCCLoss(T=T),
        criterion__reg=reg,
        **kwargs,
    )
    return net
