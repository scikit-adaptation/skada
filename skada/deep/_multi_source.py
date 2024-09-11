# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#
# License: BSD 3-Clause
import torch
import torch.nn as nn

from skada.deep.base import (
    BaseDALoss,
    DomainAwareCriterion,
    DomainAwareModule,
    DomainAwareNet,
    DomainBalancedDataLoader,
)
from skada.deep.losses import mmd_loss


class SelectDomainModule(torch.nn.Module):
    """Select domain module"""

    def __init__(self):
        super().__init__()

    def forward(self, X, sample_domain=None, is_source=True):
        if is_source:
            X = X[sample_domain, torch.arange(X.size(1))]
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
                # Doing that means that the initialization is
                # the same for aller the specific layers
                self.add_module(
                    f"layer_{i}", nn.ModuleList(layer for _ in range(n_domains))
                )
                self.add_module(f"output_layer_{i}", SelectDomainModule())
            else:
                self.add_module(f"layer_{i}", layer)
        self.n_domains = n_domains

    def forward(self, X, sample_domain=None, sample_weight=None, is_source=True):
        # if is_source:
        #     domain_present = torch.unique(sample_domain)
        #     dict_idx = {int(domain): idx for idx, domain in enumerate(domain_present)}
        #     sample_domain_ = torch.tensor(
        #         [dict_idx[int(domain)] for domain in sample_domain]
        #     )
        for i, layer in enumerate(self.children()):
            if isinstance(layer, nn.ModuleList):
                X = [layer[j](X) for j in range(self.n_domains)]
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
    """Loss MFSAN"""

    def __init__(
        self,
        sigmas=None,
    ):
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
        n_domains = len(features_t)
        mmd = 0
        disc = 0
        for i in range(n_domains):
            mmd += mmd_loss(
                features_s[torch.where(sample_domain == i)[0]],
                features_t[i],
                sigmas=self.sigmas,
            )
            for j in range(i + 1, n_domains):
                disc += torch.mean(
                    y_pred_t[i] - y_pred_t[j],
                )
        mmd /= n_domains
        disc /= n_domains * (n_domains - 1) / 2

        loss = mmd + disc
        return loss


def MFSAN(module, layer_name, reg=1, sigmas=None, base_criterion=None, **kwargs):
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
        module__is_multi_source=True,
        iterator_train=DomainBalancedDataLoader,
        criterion=DomainAwareCriterion,
        criterion__base_criterion=base_criterion,
        criterion__reg=reg,
        criterion__adapt_criterion=MFSANLoss(sigmas=sigmas),
        criterion__is_multi_source=True,
        **kwargs,
    )
    return net
