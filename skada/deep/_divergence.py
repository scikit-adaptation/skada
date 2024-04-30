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
    DomainBalancedDataLoader,
)

from .losses import dan_loss, deepcoral_loss


class DeepCoralLoss(BaseDALoss):
    """Loss DeepCORAL

    This loss reduces the distance between covariances
    of the source features and the target features.
    See [12]_.


    References
    ----------
    .. [12]  Baochen Sun and Kate Saenko. Deep coral:
            Correlation alignment for deep domain
            adaptation. In ECCV Workshops, 2016.
    """

    def __init__(
        self,
    ):
        super().__init__()

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
        loss = deepcoral_loss(features_s, features_t)
        return loss


def DeepCoral(module, layer_name, reg=1, **kwargs):
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
        The regularization parameter of the covariance estimator.

    References
    ----------
    .. [12]  Baochen Sun and Kate Saenko. Deep coral:
            Correlation alignment for deep domain
            adaptation. In ECCV Workshops, 2016.
    """
    net = DomainAwareNet(
        module=DomainAwareModule,
        module__base_module=module,
        module__layer_name=layer_name,
        iterator_train=DomainBalancedDataLoader,
        criterion=DomainAwareCriterion,
        criterion__criterion=torch.nn.CrossEntropyLoss(),
        criterion__reg=reg,
        criterion__adapt_criterion=DeepCoralLoss(),
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
    ):
        """Compute the domain adaptation loss"""
        loss = dan_loss(features_s, features_t, sigmas=self.sigmas)
        return loss


def DAN(module, layer_name, reg=1, sigmas=None, **kwargs):
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
        The regularization parameter of the covariance estimator.
    sigmas : array-like, optional (default=None)
        The sigmas for the Gaussian kernel.

    References
    ----------
    .. [14]  Mingsheng Long et. al. Learning Transferable
            Features with Deep Adaptation Networks.
            In ICML, 2015.
    """
    net = DomainAwareNet(
        module=DomainAwareModule,
        module__base_module=module,
        module__layer_name=layer_name,
        iterator_train=DomainBalancedDataLoader,
        criterion=DomainAwareCriterion(
            torch.nn.CrossEntropyLoss(), DANLoss(sigmas=sigmas), reg=reg
        ),
        criterion__criterion=torch.nn.CrossEntropyLoss(),
        criterion__reg=reg,
        criterion__adapt_criterion=DANLoss(sigmas=sigmas),
        **kwargs,
    )
    return net
