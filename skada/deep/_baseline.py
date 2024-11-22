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
    DomainOnlyDataLoader,
)


class DummyLoss(BaseDALoss):
    """Dummy.

    This loss computes nothing.
    """

    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self,
        **kwargs,
    ):
        """Compute the domain adaptation loss"""
        return 0


def SourceOnly(module, layer_name=None, base_criterion=None, **kwargs):
    """Source only method.

    Parameters
    ----------
    module : torch module (class or instance)
        A PyTorch :class:`~torch.nn.Module`.
    layer_name : Ignored
        Not used, present here for API consistency by convention.
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
        iterator_train=DomainOnlyDataLoader,
        iterator_train__domain_used="source",
        criterion=DomainAwareCriterion,
        criterion__base_criterion=base_criterion,
        criterion__reg=0,
        criterion__adapt_criterion=DummyLoss(),
        **kwargs,
    )
    return net


def TargetOnly(module, layer_name=None, base_criterion=None, **kwargs):
    """Target only method.

    Parameters
    ----------
    module : torch module (class or instance)
        A PyTorch :class:`~torch.nn.Module`.
    layer_name : Ignored
        Not used, present here for API consistency by convention.
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
        iterator_train=DomainOnlyDataLoader,
        iterator_train__domain_used="target",
        criterion=DomainAwareCriterion,
        criterion__base_criterion=base_criterion,
        criterion__train_on_target=True,
        criterion__reg=0,
        criterion__adapt_criterion=DummyLoss(),
        **kwargs,
    )
    return net
