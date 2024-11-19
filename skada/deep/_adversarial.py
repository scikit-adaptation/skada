# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Ambroise Odonnat <ambroiseodonnattechnologie@gmail.com>
#         Antoine Collas <contact@antoinecollas.fr>
#
# License: BSD 3-Clause
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from skada.deep.base import (
    BaseDALoss,
    DomainAwareCriterion,
    DomainAwareModule,
    DomainAwareNet,
    DomainBalancedDataLoader,
)

from .modules import DomainClassifier
from .utils import check_generator


class DANNLoss(BaseDALoss):
    """Loss DANN.

    This loss tries to minimize the divergence between features with
    adversarial method. The weights are updated to make harder
    to classify domains (i.e., remove domain-specific features).

    See [15]_ for details.

    Parameters
    ----------
    domain_criterion : torch criterion (class), default=None
        The initialized criterion (loss) used to compute the
        DANN loss. If None, a BCELoss is used.

    References
    ----------
    .. [15] Yaroslav Ganin et. al. Domain-Adversarial Training
            of Neural Networks. In Journal of Machine Learning
            Research, 2016.
    """

    def __init__(self, domain_criterion=None):
        super().__init__()
        if domain_criterion is None:
            self.domain_criterion_ = torch.nn.BCELoss()
        else:
            self.domain_criterion_ = domain_criterion

    def forward(
        self,
        domain_pred_s,
        domain_pred_t,
        **kwargs,
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

        return loss


def DANN(
    module,
    layer_name,
    reg=1,
    domain_classifier=None,
    num_features=None,
    base_criterion=None,
    domain_criterion=None,
    **kwargs,
):
    """Domain-Adversarial Training of Neural Networks (DANN).

    From [15]_.

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
        Regularization parameter for DA loss.
    domain_classifier : torch module, default=None
        A PyTorch :class:`~torch.nn.Module` used to classify the
        domain. If None, a domain classifier is created following [1]_.
    num_features : int, default=None
        Size of the input of domain classifier,
        e.g size of the last layer of
        the feature extractor.
        If domain_classifier is None, num_features has to be
        provided.
    base_criterion : torch criterion (class)
        The base criterion used to compute the loss with source
        labels. If None, the default is `torch.nn.CrossEntropyLoss`.
    domain_criterion : torch criterion (class)
        The criterion (loss) used to compute the
        DANN loss. If None, a BCELoss is used.

    References
    ----------
    .. [15] Yaroslav Ganin et. al. Domain-Adversarial Training
            of Neural Networks. In Journal of Machine Learning
            Research, 2016.
    """
    if domain_classifier is None:
        # raise error if num_feature is None
        if num_features is None:
            raise ValueError(
                "If domain_classifier is None, num_features has to be provided"
            )
        domain_classifier = DomainClassifier(num_features=num_features)

    if base_criterion is None:
        base_criterion = torch.nn.CrossEntropyLoss()

    net = DomainAwareNet(
        module=DomainAwareModule,
        module__base_module=module,
        module__layer_name=layer_name,
        module__domain_classifier=domain_classifier,
        iterator_train=DomainBalancedDataLoader,
        criterion=DomainAwareCriterion,
        criterion__base_criterion=base_criterion,
        criterion__reg=reg,
        criterion__adapt_criterion=DANNLoss(domain_criterion=domain_criterion),
        **kwargs,
    )
    return net


class CDANLoss(BaseDALoss):
    """Conditional Domain Adversarial Networks (CDAN) loss.

    This loss tries to minimize the divergence between features with
    adversarial method. The weights are updated to make harder
    to classify domains (i.e., remove domain-specific features)
    via multilinear conditioning that captures the crosscovariance between
    feature representations and classifier predictions
    From [16]_.

    Parameters
    ----------
    domain_criterion : torch criterion (class), default=None
        The initialized criterion (loss) used to compute the
        CDAN loss. If None, a BCELoss is used.

    References
    ----------
    .. [16] Mingsheng Long et. al. Conditional Adversarial Domain Adaptation
            In NeurIPS, 2016.
    """

    def __init__(self, domain_criterion=None):
        super().__init__()
        if domain_criterion is None:
            self.domain_criterion_ = torch.nn.BCELoss()
        else:
            self.domain_criterion_ = domain_criterion

    def forward(
        self,
        domain_pred_s,
        domain_pred_t,
        features_s,
        features_t,
        **kwargs,
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
        return loss


class CDANModule(DomainAwareModule):
    """Conditional Domain Adversarial Networks (CDAN) module.

    From [16]_.

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
    .. [16] Mingsheng Long et. al. Conditional Adversarial Domain Adaptation
            In NeurIPS, 2016.
    """

    def __init__(
        self,
        base_module,
        layer_name,
        domain_classifier,
        max_features=4096,
        random_state=42,
    ):
        super().__init__(base_module, layer_name, domain_classifier)
        self.max_features = max_features
        self.random_state = random_state

    def forward(
        self,
        X,
        sample_domain=None,
        sample_idx=None,
        is_fit=False,
        return_features=False,
    ):
        if is_fit:
            # predict
            y_pred = self.base_module_(X)
            features = self.intermediate_layers[self.layer_name]

            n_classes = y_pred.shape[1]
            n_features = features.shape[1]
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
                multilinear_map = torch.bmm(y_pred.unsqueeze(2), features.unsqueeze(1))
                multilinear_map = multilinear_map.view(-1, n_features * n_classes)

            else:
                multilinear_map = random_layer.forward([features, y_pred])

            domain_pred = self.domain_classifier_(multilinear_map)

            return (
                y_pred,
                domain_pred,
                features,
                sample_domain,
                sample_idx,
            )
        else:
            if return_features:
                return self.base_module_(X), self.intermediate_layers[self.layer_name]
            else:
                return self.base_module_(X)


def CDAN(
    module,
    layer_name,
    reg=1,
    max_features=4096,
    domain_classifier=None,
    num_features=None,
    n_classes=None,
    base_criterion=None,
    domain_criterion=None,
    **kwargs,
):
    """Conditional Domain Adversarial Networks (CDAN).

    From [16]_.

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
        Regularization parameter for DA loss.
    max_features : int, default=4096
        Maximum size of the input for the domain classifier.
        4096 is the largest number of units in typical deep network
        according to [1]_.
    domain_classifier : torch module, default=None
        A PyTorch :class:`~torch.nn.Module` used to classify the
        domain. If None, a domain classifier is created following [1]_.
    num_features : int, default=None
        Size of the embedding space e.g. the size of the output of layer_name.
        If domain_classifier is None, num_features has to be
        provided.
    n_classes : int, default None
        Number of output classes.
        If domain_classifier is None, n_classes has to be provided.
    base_criterion : torch criterion (class)
        The base criterion used to compute the loss with source
        labels. If None, the default is `torch.nn.CrossEntropyLoss`.
    domain_criterion : torch criterion (class)
        The criterion (loss) used to compute the
        CDAN loss. If None, a BCELoss is used.

    References
    ----------
    .. [16]  Mingsheng Long et. al. Conditional Adversarial Domain Adaptation
            In NeurIPS, 2016.
    """
    if domain_classifier is None:
        if num_features is None:
            raise ValueError(
                "If domain_classifier is None, num_features has to be provided"
            )
        if n_classes is None:
            raise ValueError(
                "If domain_classifier is None, n_classes has to be provided"
            )
        num_features = np.min([num_features * n_classes, max_features])
        domain_classifier = DomainClassifier(num_features=num_features)

    if base_criterion is None:
        base_criterion = torch.nn.CrossEntropyLoss()

    net = DomainAwareNet(
        module=CDANModule,
        module__base_module=module,
        module__layer_name=layer_name,
        module__domain_classifier=domain_classifier,
        module__max_features=max_features,
        iterator_train=DomainBalancedDataLoader,
        criterion=DomainAwareCriterion,
        criterion__base_criterion=base_criterion,
        criterion__reg=reg,
        criterion__adapt_criterion=CDANLoss(domain_criterion=domain_criterion),
        **kwargs,
    )
    return net


class ModifiedCrossEntropyLoss(torch.nn.Module):
    """Modified CrossEntropyLoss as described in (29) from [35]_ with label smoothing.

    Parameters
    ----------
    smoothing : float, default=0.1
        Smoothing factor for label smoothing.
    """

    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        """
        Compute the modified CrossEntropyLoss
        with label smoothing applied to predictions.

        Parameters
        ----------
        input : torch.Tensor
            Predictions from the model.
        target : torch.Tensor
            Target labels.

        Returns
        -------
        loss : torch.Tensor
            The modified CrossEntropyLoss with label smoothing applied to predictions.
        """
        # Compute probabilities
        prob = F.softmax(input, dim=-1)

        # Apply label smoothing to predictions
        num_classes = input.shape[-1]
        smooth_prob = (1 - self.smoothing) * prob + self.smoothing / num_classes

        # Gather probabilities of the target classes
        target_prob = smooth_prob[..., target]

        # Compute log(1 - probability) for the target class
        log_one_minus_prob = torch.log(1 - target_prob)

        # Return mean loss
        return torch.mean(log_one_minus_prob)


class MDDLoss(BaseDALoss):
    """Loss MDD.

    This loss tries to minimize the disparity discrepancy between
    the source and target domains. The discrepancy is estimated
    through adversarial training of three networks: an encoder,
    a task network and a discriminator.

    See [35]_ for details.

    Parameters
    ----------
    gamma : float (default=4.0)
        Margin parameter following [35]_

    References
    ----------
    .. [35] Yuchen Zhang et. al. Bridging Theory and Algorithm
            for Domain Adaptation. In International Conference on
            Machine Learning, 2019.
    """

    def __init__(self, gamma=4.0):
        super().__init__()
        self.gamma = gamma
        self.disc_criterion_s = torch.nn.CrossEntropyLoss()
        self.disc_criterion_t = ModifiedCrossEntropyLoss()

    def forward(
        self,
        y_pred_s,
        y_pred_t,
        domain_pred_s,
        domain_pred_t,
        **kwargs,
    ):
        """Compute the domain adaptation loss"""
        # TODO: handle binary classification
        # Multiclass classification
        pseudo_label_s = torch.argmax(y_pred_s, axis=-1)
        pseudo_label_t = torch.argmax(y_pred_t, axis=-1)

        disc_loss_s = self.disc_criterion_s(domain_pred_s, pseudo_label_s)
        disc_loss_t = self.disc_criterion_t(domain_pred_t, pseudo_label_t)

        # Compute the MDD loss value
        disc_loss = self.gamma * disc_loss_s - disc_loss_t

        return disc_loss


def MDD(
    module,
    layer_name,
    reg=1,
    gamma=4.0,
    disc_classifier=None,
    num_features=None,
    n_classes=None,
    base_criterion=None,
    **kwargs,
):
    """Margin Disparity Discrepancy (MDD).

    From [35]_.

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
        Regularization parameter for DA loss.
    disc_classifier : torch module, default=None
        A PyTorch :class:`~torch.nn.Module` used as a discriminator.
        It should have the same architecture than the classifier
        used on the source. If None, a domain classifier is
        created following [1]_.
    num_features : int, default=None
        Size of the input of domain classifier,
        e.g size of the last layer of
        the feature extractor.
        If domain_classifier is None, num_features has to be
        provided.
    n_classes : int, default=None
        Number of classes. If domain_classifier is None,
        n_classes has to be provided.
    base_criterion : torch criterion (class)
        The base criterion used to compute the loss with source
        labels. If None, the default is `torch.nn.CrossEntropyLoss`.
    gamma : float (default=4.0)
        Margin parameter following [35]_.

    References
    ----------
    .. [35] Yuchen Zhang et. al. Bridging Theory and Algorithm
            for Domain Adaptation. In International Conference on
            Machine Learning, 2019.
    """
    if disc_classifier is None:
        # raise error if num_feature is None
        if num_features is None:
            raise ValueError(
                "If disc_classifier is None, num_features has to be provided"
            )
        disc_classifier = DomainClassifier(
            num_features=num_features, n_classes=n_classes
        )

    if base_criterion is None:
        base_criterion = torch.nn.CrossEntropyLoss()

    net = DomainAwareNet(
        module=DomainAwareModule,
        module__base_module=module,
        module__layer_name=layer_name,
        module__domain_classifier=disc_classifier,
        iterator_train=DomainBalancedDataLoader,
        criterion=DomainAwareCriterion,
        criterion__base_criterion=base_criterion,
        criterion__reg=reg,
        criterion__adapt_criterion=MDDLoss(
            gamma=gamma,
        ),
        **kwargs,
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
        super().__init__()
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
