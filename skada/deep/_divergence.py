# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#
# License: BSD 3-Clause
import torch

from skada.deep.base import (
    BaseDALoss,
    DomainAwareCriterion,
    DomainAwareModule,
    DomainAwareNet,
)
from skada.deep.dataloaders import DomainBalancedDataLoader

from .losses import deepcoral_loss, mmd_loss


class DeepCoralLoss(BaseDALoss):
    """Loss DeepCORAL

    This loss reduces the distance between covariances
    of the source features and the target features.
    See [12]_.

    Parameters
    ----------
    assume_centered: bool, default=False
        If True, data are not centered before computation.

    References
    ----------
    .. [12]  Baochen Sun and Kate Saenko. Deep coral:
            Correlation alignment for deep domain
            adaptation. In ECCV Workshops, 2016.
    """

    def __init__(
        self,
        assume_centered=False,
    ):
        super().__init__()
        self.assume_centered = assume_centered

    def forward(
        self,
        y_s,
        y_pred_s,
        y_pred_t,
        domain_pred_s,
        domain_pred_t,
        features_s,
        features_t,
        sample_domain,
    ):
        """Compute the domain adaptation loss"""
        loss = deepcoral_loss(features_s, features_t, self.assume_centered)
        return loss


def DeepCoral(
    module, layer_name, reg=1, assume_centered=False, base_criterion=None, **kwargs
):
    """DeepCORAL domain adaptation method.

    From [12]_.

    Parameters
    ----------
    module : torch module (class or instance)
        A PyTorch :class:`~torch.nn.Module`.
    layer_name : str
        The name of the module's layer whose outputs are
        collected during the training for the adaptation.
    reg : float, optional (default=1)
        Regularization parameter for DA loss.
    assume_centered: bool, default=False
        If True, data are not centered before computation.
    base_criterion : torch criterion (class)
        The base criterion used to compute the loss with source
        labels. If None, the default is `torch.nn.CrossEntropyLoss`.

    References
    ----------
    .. [12]  Baochen Sun and Kate Saenko. Deep coral:
            Correlation alignment for deep domain
            adaptation. In ECCV Workshops, 2016.
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
        criterion__reg=reg,
        criterion__adapt_criterion=DeepCoralLoss(assume_centered=assume_centered),
        **kwargs,
    )
    return net


class DANLoss(BaseDALoss):
    """Loss DAN

    This loss reduces the MMD distance between
    source features and target features.
    From [14]_.

    Parameters
    ----------
    sigmas : array-like, optional (default=None)
        The sigmas for the Gaussian kernel.

    References
    ----------
    .. [14]  Mingsheng Long et. al. Learning Transferable
            Features with Deep Adaptation Networks.
            In ICML, 2015.
    """

    def __init__(self, sigmas=None):
        super().__init__()
        self.sigmas = sigmas

    def forward(
        self,
        y_s,
        y_pred_s,
        y_pred_t,
        domain_pred_s,
        domain_pred_t,
        features_s,
        features_t,
        sample_domain,
    ):
        """Compute the domain adaptation loss"""
        loss = mmd_loss(features_s, features_t, sigmas=self.sigmas)
        return loss


def DAN(module, layer_name, reg=1, sigmas=None, base_criterion=None, **kwargs):
    """DAN domain adaptation method.

    See [14]_.

    Parameters
    ----------
    module : torch module (class or instance)
        A PyTorch :class:`~torch.nn.Module`.
    layer_name : str
        The name of the module's layer whose outputs are
        collected during the training for the adaptation.
    reg : float, optional (default=1)
        Regularization parameter for DA loss.
    sigmas : array-like, optional (default=None)
        The sigmas for the Gaussian kernel.
    base_criterion : torch criterion (class)
        The base criterion used to compute the loss with source
        labels. If None, the default is `torch.nn.CrossEntropyLoss`.

    References
    ----------
    .. [14]  Mingsheng Long et. al. Learning Transferable
            Features with Deep Adaptation Networks.
            In ICML, 2015.
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
        criterion__reg=reg,
        criterion__adapt_criterion=DANLoss(sigmas=sigmas),
        **kwargs,
    )
    return net
