# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#
# License: BSD 3-Clause
import torch
from skada.deep.base import (
    DomainAwareModule,
    DomainAwareCriterion,
    DomainBalancedDataLoader,
    DomainAwareNet,
    BaseDALoss,
)
from .losses import deepcoral_loss, dan_loss


class DeepCoralLoss(BaseDALoss):
    """Loss DeepCORAL

    From [1]_.


    References
    ----------
    .. [1]  Baochen Sun and Kate Saenko. Deep coral:
            Correlation alignment for deep domain
            adaptation. In ECCV Workshops, 2016.
    """
    def __init__(self,):
        super(DeepCoralLoss, self).__init__()

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
        loss = deepcoral_loss(cov_s, cov_t)
        return loss


def DeepCoral(module, layer_name, reg=1, **kwargs):
    """DeepCORAL domain adaptation method.

    From [1]_.

    Parameters
    ----------
    module : torch module (class or instance)
        A PyTorch :class:`~torch.nn.Module`.
    layer_name : str
        The name of the module's layer whose outputs are
        collected during the training for the adaptation.
    reg : float, optional (default=1)
        The regularization parameter of the covariance estimator.

    References
    ----------
    .. [1]  Baochen Sun and Kate Saenko. Deep coral:
            Correlation alignment for deep domain
            adaptation. In ECCV Workshops, 2016.
    """
    net = DomainAwareNet(
        DomainAwareModule(module, layer_name),
        iterator_train=DomainBalancedDataLoader,
        criterion=DomainAwareCriterion(
            torch.nn.CrossEntropyLoss(), DeepCoralLoss(), reg=reg
        ),
        **kwargs
    )
    return net


class DANLoss(BaseDALoss):
    """Loss DAN

    From [1]_.


    References
    ----------
    .. [1]  Mingsheng Long et. al. Learning Transferable
            Features with Deep Adaptation Networks.
            In ICML, 2015.
    """
    def __init__(self,):
        super(DANLoss, self).__init__()

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
        loss = dan_loss(features_s, features_t)
        return loss


def DAN(module, layer_name, reg=1, **kwargs):
    """DAN domain adaptation method.

    From [1]_.

    Parameters
    ----------
    module : torch module (class or instance)
        A PyTorch :class:`~torch.nn.Module`.
    layer_name : str
        The name of the module's layer whose outputs are
        collected during the training for the adaptation.
    reg : float, optional (default=1)
        The regularization parameter of the covariance estimator.

    References
    ----------
    .. [1]  Mingsheng Long et. al. Learning Transferable
            Features with Deep Adaptation Networks.
            In ICML, 2015.
    """
    net = DomainAwareNet(
        DomainAwareModule(module, layer_name),
        iterator_train=DomainBalancedDataLoader,
        criterion=DomainAwareCriterion(
            torch.nn.CrossEntropyLoss(), DANLoss(), reg=reg
        ),
        **kwargs
    )
    return net
