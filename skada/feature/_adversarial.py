# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#
# License: BSD 3-Clause
import math

import torch
from torch import nn

from skada.feature.base import (
    DomainAwareCriterion,
    DomainBalancedDataLoader,
    DomainAwareNet,
    BaseDALoss,
)

from .utils import check_generator


class DANNLoss(nn.Module):
    """Loss DANN.

    See [1]_.

    Parameters
    ----------
    reg : float, default=1
        Regularization parameter.
    target_criterion : torch criterion (class)
        The uninitialized criterion (loss) used to compute the
        DANN loss. The criterion should support reduction='none'.

    References
    ----------
    .. [1]  Yaroslav Ganin et. al. Domain-Adversarial Training
            of Neural Networks  In Journal of Machine Learning
            Research, 2016.
    """

    def __init__(self, reg=1, domain_criterion=None):
        super(DANNLoss, self).__init__()
        self.reg = reg
        if domain_criterion is None:
            self.domain_criterion_ = torch.nn.BCELoss()
        else:
            self.domain_criterion_ = domain_criterion

    def forward(
        self,
        y_s,
        y_pred_t,
        y_pred_domain_s,
        y_pred_domain_t,
        features_s,
        features_t,
    ):
        """Compute the domain adaptation loss"""
        domain_label = torch.zeros(
            (features_s.size()[0]), device=self.device,
        )
        domain_label_target = torch.ones(
            (features_t.size()[0]), device=self.device,
        )

        # update classification function
        loss = (
            self.domain_criterion_(y_pred_domain_s, domain_label) +
            self.domain_criterion_(y_pred_domain_t, domain_label_target)
        )
        return self.reg * loss


def DANN(module, layer_name, reg=1, **kwargs):
    """Domain-Adversarial Training of Neural Networks (DANN).

    From [1]_.

    Parameters
    ----------
    module : torch module (class or instance)
        A PyTorch :class:`~torch.nn.Module`. In general, the
        uninstantiated class should be passed, although instantiated
        modules will also work.
    layer_name : str
        The name of the module's layer whose outputs are
        collected during the training.
    reg : float, default=1
        Regularization parameter.

    References
    ----------
    .. [1]  Yaroslav Ganin et. al. Domain-Adversarial Training
            of Neural Networks  In Journal of Machine Learning
            Research, 2016.
    """
    net = DomainAwareNet(
        module,
        layer_name,
        iterator_train=DomainBalancedDataLoader,
        criterion=DomainAwareCriterion(
            torch.nn.CrossEntropyLoss(), DANNLoss(reg=reg)
        ),
        **kwargs
    )
    return net


class CDANLoss(BaseDALoss):
    """Conditional Domain Adversarial Networks (CDAN) loss.

    From [1]_.

    Parameters
    ----------
    reg : float, default=1
        Regularization parameter.
    max_features : int, default=4096
        Maximum size of the input for the domain classifier.
        4096 is the largest number of units in typical deep network
        according to [1]_.
    target_criterion : torch criterion (class)
        The uninitialized criterion (loss) used to compute the
        CDAN loss. The criterion should support reduction='none'.

    References
    ----------
    .. [1]  Mingsheng Long et. al. Conditional Adversarial Domain Adaptation
            In NeurIPS, 2016.
    """

    def __init__(self, reg=1, max_features=4096, domain_criterion=None):
        super(CDANLoss, self).__init__()
        self.reg = reg
        self.max_features = max_features
        if domain_criterion is None:
            self.domain_criterion_ = torch.nn.BCELoss()
        else:
            self.domain_criterion_ = domain_criterion

    def forward(
        self,
        y_s,
        y_pred_t,
        y_pred_domain_s,
        y_pred_domain_t,
        features_s,
        features_t,
    ):
        """Compute the domain adaptation loss"""
        dtype = torch.float32
        n_classes = y_s.shape[1]
        n_features = features_s.shape[1]
        if n_features * n_classes > self.max_features:
            random_layer = _RandomLayer(
                self.random_state,
                input_dims=[n_features, n_classes],
                output_dim=self.max_features
            )
        else:
            random_layer = None

        # Compute the input for the domain classifier
        if random_layer is None:
            multilinear_map = torch.bmm(
                y_s.unsqueeze(2), features_s.unsqueeze(1)
            )
            multilinear_map_target = torch.bmm(
                y_s.unsqueeze(2), features_t.unsqueeze(1)
            )

            multilinear_map = multilinear_map.view(-1, n_features * n_classes)
            multilinear_map_target = multilinear_map_target.view(
                    -1, n_features * n_classes)

        else:
            multilinear_map = random_layer.forward([features_s, y_s])
            multilinear_map_target = random_layer.forward(
                [features_t, y_pred_t]
            )

        # Compute the output of the domain classifier
        output_domain = self.domain_classifier_.forward(
            multilinear_map, self.reg
        ).flatten()
        output_domain_target = self.domain_classifier_.forward(
            multilinear_map_target, self.reg
        ).flatten()

        # create domain label
        domain_label = torch.zeros(
            (features_s.size()[0]), device=self.device, dtype=dtype
        )
        domain_label_target = torch.ones(
            (features_t.size()[0]), device=self.device, dtype=dtype
        )

        # update classification function
        loss = (
            self.domain_criterion_(output_domain, domain_label) +
            self.domain_criterion_(output_domain_target, domain_label_target)
        )
        return self.reg * loss


def CDAN(module, layer_name, reg=1, max_features=4096, **kwargs):
    """Conditional Domain Adversarial Networks (CDAN).

    From [1]_.

    Parameters
    ----------
    module : torch module (class or instance)
        A PyTorch :class:`~torch.nn.Module`. In general, the
        uninstantiated class should be passed, although instantiated
        modules will also work.
    layer_name : str
        The name of the module's layer whose outputs are
        collected during the training.
    reg : float, default=1
        Regularization parameter.
    max_features : int, default=4096
        Maximum size of the input for the domain classifier.
        4096 is the largest number of units in typical deep network
        according to [1]_.

    References
    ----------
    .. [1]  Mingsheng Long et. al. Conditional Adversarial Domain Adaptation
            In NeurIPS, 2016.
    """
    net = DomainAwareNet(
        module,
        layer_name,
        iterator_train=DomainBalancedDataLoader,
        criterion=DomainAwareCriterion(
            torch.nn.CrossEntropyLoss(), CDANLoss(reg=reg, max_features=max_features)
        ),
        **kwargs
    )
    return net


class _RandomLayer(nn.Module):
    """Randomized multilinear map layer.

    Parameters
    ----------
    random_state : int, Generator instance or None
        Determines random number generation for random layer creation
    input_dims : list of int
        List of input dimensions.
    output_dims : int
        Output dimension wanted.
    """

    def __init__(self, random_state, input_dims, output_dim=4096):
        super(_RandomLayer, self).__init__()
        gen = check_generator(random_state)
        self.output_dim = output_dim
        self.random_matrix = [
            torch.randn(size=(input_dims[i], output_dim), generator=gen)
            for i in range(len(input_dims))
        ]

    def forward(self, input_list):
        device = input_list[0].device
        return_list = [
            torch.mm(input_list[i], self.random_matrix[i].to(device))
            for i in range(len(input_list))
        ]
        return_tensor = (
            return_list[0] / math.pow(float(self.output_dim), 1.0/len(input_list))
        )
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor
