# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#
# License: BSD 3-Clause
import torch
import torch.nn as nn
import torch.nn.functional as F

from skada.deep.base import (
    BaseDALoss,
    DomainAwareCriterion,
    DomainAwareModule,
    DomainAwareNet,
)
from skada.deep.dataloaders import MultiSourceDomainBalancedDataLoader
from skada.deep.losses import mmd_loss


class SelectDomainModule(torch.nn.Module):
    """Select domain module"""

    def __init__(self):
        super().__init__()

    def forward(self, X, sample_domain=None, is_source=True):
        if is_source:
            X = X[sample_domain - 1, torch.arange(X.size(1))]
        return X


class MultiSourceModule(torch.nn.Module):
    """Multi-source module

    A multi-source module allowing domain-specific layers

    Parameters
    ----------
    layers : list of torch modules (list)
       list of the Module in the order.
    domain_specific_layers : dict
        A list of True or False saying if the layer should domain-specific or not.
    n_source_domains : int
        The number of domains.
    """

    def __init__(self, layers, domain_specific_layers, n_domains):
        super().__init__()
        for i, layer in enumerate(layers):
            if domain_specific_layers[i]:
                self.add_module(
                    f"layer_{i}", nn.ModuleList(layer for _ in range(n_domains))
                )
                self.add_module(f"output_layer_{i}", SelectDomainModule())
            else:
                self.add_module(f"layer_{i}", layer)
        self.n_domains = n_domains

    def forward(self, X, sample_domain=None, sample_weight=None, is_source=True):
        for i, layer in enumerate(self.children()):
            if isinstance(layer, nn.ModuleList):
                if X.size(0) != self.n_domains:
                    X = [layer[j](X) for j in range(self.n_domains)]
                else:
                    X = [layer[j](X[j]) for j in range(self.n_domains)]
                X = torch.stack(X, dim=0)
            elif isinstance(layer, SelectDomainModule):
                if is_source:
                    X = layer(X, sample_domain)
                else:
                    X = layer(X, is_source=is_source)
            else:
                X = layer(X)
        return X


class MFSANLoss(BaseDALoss):
    """Loss MFSAN

    The loss for the MFSAN method proposed in [33].


    Parameters
    ----------
    reg_mmd : float, optional (default=1)
        The regularization parameter of the MMD loss.
    reg_cl : float, optional (default=1)
        The regularization parameter of the target discrepancy
        classification loss.
    sigmas : array-like, optional (default=None)
        The sigmas for the Gaussian kernel.

    References
    ----------
    .. [33] Zhu, Y., Zhuang, F., and Wang, D., (2022).
            Aligning Domain-specific Distribution and Classifier
            for Cross-domain Classification from Multiple Sources.
            Association for the Advancement of Artificial Intelligence.
    """

    def __init__(
        self,
        reg_mmd=1,
        reg_cl=1,
        sigmas=None,
    ):
        super().__init__()
        self.reg_mmd = reg_mmd
        self.reg_cl = reg_cl
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
        n_domains = len(features_t)
        source_idx = sample_domain > 0
        mmd = 0
        disc = 0
        for i in range(n_domains):
            mmd += mmd_loss(
                features_s[torch.where(sample_domain[source_idx] == i + 1)[0]],
                features_t[i],
                sigmas=self.sigmas,
            )
            for j in range(i + 1, n_domains):
                disc += torch.mean(
                    torch.abs(F.softmax(y_pred_t[i]) - F.softmax(y_pred_t[j])),
                )
        mmd /= n_domains
        disc /= n_domains * (n_domains - 1) / 2
        loss = self.reg_mmd * mmd + self.reg_cl * disc
        return loss


def MFSAN(
    module,
    layer_name,
    source_domains,
    reg_mmd=1,
    reg_cl=1,
    sigmas=None,
    base_criterion=None,
    **kwargs,
):
    """MFSAN domain adaptation method.

    See [33]_.

    Parameters
    ----------
    module : torch module (class or instance)
        A PyTorch :class:`~torch.nn.Module`.
    layer_name : str
        The name of the module's layer whose outputs are
        collected during the training for the adaptation.
    source_domains : list of int
        The list of source domains.
    reg_mmd : float, optional (default=1)
        The regularization parameter of the MMD loss.
    reg_cl : float, optional (default=1)
        The regularization parameter of the target discrepancy
        classification loss.
    sigmas : array-like, optional (default=None)
        The sigmas for the Gaussian kernel.
    base_criterion : torch criterion (class)
        The base criterion used to compute the loss with source
        labels. If None, the default is `torch.nn.CrossEntropyLoss`.

    References
    ----------
    .. [33] Zhu, Y., Zhuang, F., and Wang, D., (2022).
            Aligning Domain-specific Distribution and Classifier
            for Cross-domain Classification from Multiple Sources.
            Association for the Advancement of Artificial Intelligence.
    """
    if base_criterion is None:
        base_criterion = torch.nn.CrossEntropyLoss()

    net = DomainAwareNet(
        module=DomainAwareModule,
        module__base_module=module,
        module__layer_name=layer_name,
        module__is_multi_source=True,
        iterator_train=MultiSourceDomainBalancedDataLoader,
        iterator_train__source_domains=source_domains,
        criterion=DomainAwareCriterion,
        criterion__base_criterion=base_criterion,
        criterion__reg=1,
        criterion__adapt_criterion=MFSANLoss(
            reg_mmd=reg_mmd, reg_cl=reg_cl, sigmas=sigmas
        ),
        criterion__is_multi_source=True,
        **kwargs,
    )
    return net
