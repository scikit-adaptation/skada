# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#
# License: BSD 3-Clause
import torch
import torch.nn as nn


class SelectDomainModule(torch.nn.Module):
    """Select domain module"""

    def __init__(self):
        super().__init__()

    def forward(self, X, sample_domain=None, is_source=True):
        if is_source:
            X = X[sample_domain]
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
        if is_source:
            domain_present = torch.unique(sample_domain)
            dict_idx = {int(domain): idx for idx, domain in enumerate(domain_present)}
            sample_domain_ = torch.tensor(
                [dict_idx[int(domain)] for domain in sample_domain]
            )
        for i, layer in enumerate(self.children()):
            if isinstance(layer, nn.ModuleList):
                X = [layer[j](X) for j in domain_present]
                X = torch.stack(X, dim=0)
            elif isinstance(layer, SelectDomainModule):
                if is_source:
                    X = layer(X, sample_domain_)
                else:
                    X = layer(X, is_source=is_source)
            else:
                X = layer(X)
        return X


class MFSANLoss(torch.nn.Module):
    """Multiple feature spaces adaptation network"""

    def __init__(self, base_module, domain_classifier):
        super().__init__()
        self.base_module_ = base_module

    def forward(self, X, sample_domain, sample_weight=None, is_fit=False):
        X = self.base_module_(X, sample_domain)
        return X
