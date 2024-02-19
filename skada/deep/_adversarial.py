# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#
# License: BSD 3-Clause
import math

import torch
from torch import nn

from skada.deep.base import (
    DomainAwareModule,
    DomainAwareCriterion,
    DomainBalancedDataLoader,
    DomainAwareNet,
    BaseDALoss,
)
from .modules import DomainClassifier

from .utils import check_generator


class DANNLoss(BaseDALoss):
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
        y_pred_s,
        y_pred_t,
        domain_pred_s,
        domain_pred_t,
        features_s,
        features_t,
    ):
        """Compute the domain adaptation loss"""
        domain_label = torch.zeros(
            (domain_pred_s.size()[0]),
            device=domain_pred_s.device,
        )
        domain_label_target = torch.ones(
            (domain_pred_t.size()[0]),
            device=domain_pred_t.device,
        )

        # update classification function
        loss = self.domain_criterion_(
            domain_pred_s, domain_label
        ) + self.domain_criterion_(domain_pred_t, domain_label_target)

        return self.reg * loss


def DANN(module, layer_name, reg=1, len_last_layer=1, domain_classifier=None, **kwargs):
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
    len_last_layer : int, default=1
        Size of the input of domain classifier,
        e.g size of the last layer of
        the feature extractor.
    domain_classifier : torch module
        A PyTorch :class:`~torch.nn.Module` used to classify the
        domain.

    References
    ----------
    .. [1]  Yaroslav Ganin et. al. Domain-Adversarial Training
            of Neural Networks  In Journal of Machine Learning
            Research, 2016.
    """
    if domain_classifier is None:
        domain_classifier = DomainClassifier(alpha=reg, len_last_layer=len_last_layer)

    net = DomainAwareNet(
        DomainAwareModule(module, layer_name, domain_classifier),
        iterator_train=DomainBalancedDataLoader,
        criterion=DomainAwareCriterion(torch.nn.CrossEntropyLoss(), DANNLoss(reg=reg)),
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
    target_criterion : torch criterion (class)
        The uninitialized criterion (loss) used to compute the
        CDAN loss. The criterion should support reduction='none'.

    References
    ----------
    .. [1]  Mingsheng Long et. al. Conditional Adversarial Domain Adaptation
            In NeurIPS, 2016.
    """

    def __init__(self, reg=1, domain_criterion=None):
        super(CDANLoss, self).__init__()
        self.reg = reg
        if domain_criterion is None:
            self.domain_criterion_ = torch.nn.BCELoss()
        else:
            self.domain_criterion_ = domain_criterion

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
        dtype = torch.float32

        # create domain label
        domain_label = torch.zeros(
            (features_s.size()[0]), device=features_s.device, dtype=dtype
        )
        domain_label_target = torch.ones(
            (features_t.size()[0]), device=features_s.device, dtype=dtype
        )

        # update classification function
        loss = self.domain_criterion_(
            domain_pred_s, domain_label
        ) + self.domain_criterion_(domain_pred_t, domain_label_target)
        return self.reg * loss


class CDANModule(DomainAwareModule):
    """Conditional Domain Adversarial Networks (CDAN) module.

    From [1]_.

    Parameters
    ----------
    module : torch module (class or instance)
        A PyTorch :class:`~torch.nn.Module`.
    layer_name : str
        The name of the module's layer whose outputs are
        collected during the training for adaptation.
    domain_classifier : torch module
        A PyTorch :class:`~torch.nn.Module` used to classify the
        domain.
    max_features : int, default=4096
        Maximum size of the input for the domain classifier.
        4096 is the largest number of units in typical deep network
        according to [1]_.

    References
    ----------
    .. [1]  Mingsheng Long et. al. Conditional Adversarial Domain Adaptation
            In NeurIPS, 2016.
    """
    def __init__(self, module, layer_name, domain_classifier, max_features=4096):
        super(CDANModule, self).__init__(module, layer_name, domain_classifier)
        self.max_features = max_features

    def forward(self, X, sample_domain, is_fit=False, return_features=False):
        if is_fit:
            X_t = X[sample_domain < 0]
            X_s = X[sample_domain > 0]
            # predict
            y_pred_s = self.module_(X_s)
            features_s = self.intermediate_layers[self.layer_name]
            y_pred_t = self.module_(X_t)
            features_t = self.intermediate_layers[self.layer_name]

            n_classes = y_pred_s.shape[1]
            n_features = features_s.shape[1]
            if n_features * n_classes > self.max_features:
                random_layer = _RandomLayer(
                    self.random_state,
                    input_dims=[n_features, n_classes],
                    output_dim=self.max_features,
                )
            else:
                random_layer = None

            # Compute the input for the domain classifier
            if random_layer is None:
                multilinear_map = torch.bmm(
                    y_pred_s.unsqueeze(2), features_s.unsqueeze(1)
                )
                multilinear_map_target = torch.bmm(
                    y_pred_t.unsqueeze(2), features_t.unsqueeze(1)
                )

                multilinear_map = multilinear_map.view(-1, n_features * n_classes)
                multilinear_map_target = multilinear_map_target.view(
                    -1, n_features * n_classes
                )

            else:
                multilinear_map = random_layer.forward([features_s, y_pred_s])
                multilinear_map_target = random_layer.forward([features_t, y_pred_t])

            if self.domain_classifier_ is not None:
                print(multilinear_map.shape)
                domain_pred_s = self.domain_classifier_(multilinear_map)
                domain_pred_t = self.domain_classifier_(multilinear_map_target)
                domain_pred = torch.empty((len(sample_domain)))
                domain_pred[sample_domain > 0] = domain_pred_s
                domain_pred[sample_domain < 0] = domain_pred_t
            else:
                domain_pred = None

            y_pred = torch.empty((len(sample_domain), y_pred_s.shape[1]))
            y_pred[sample_domain > 0] = y_pred_s
            y_pred[sample_domain < 0] = y_pred_t

            features = torch.empty((len(sample_domain), features_s.shape[1]))
            features[sample_domain > 0] = features_s
            features[sample_domain < 0] = features_t

            return (
                y_pred,
                domain_pred,
                features,
                sample_domain,
            )
        else:
            if return_features:
                return self.module_(X), self.intermediate_layers[self.layer_name]
            else:
                return self.module_(X)


def CDAN(
    module, layer_name, reg=1, max_features=4096, domain_classifier=None, **kwargs
):
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
    domain_classifier : torch module
        A PyTorch :class:`~torch.nn.Module` used to classify the
        domain.

    References
    ----------
    .. [1]  Mingsheng Long et. al. Conditional Adversarial Domain Adaptation
            In NeurIPS, 2016.
    """
    if domain_classifier is None:
        domain_classifier = DomainClassifier(alpha=reg)

    net = DomainAwareNet(
        CDANModule(module, layer_name, domain_classifier, max_features=max_features),
        iterator_train=DomainBalancedDataLoader,
        criterion=DomainAwareCriterion(
            torch.nn.CrossEntropyLoss(), CDANLoss(reg=reg)
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
        return_tensor = return_list[0] / math.pow(
            float(self.output_dim), 1.0 / len(input_list)
        )
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor
