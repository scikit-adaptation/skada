# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD 3-Clause
import math

import torch
from torch import nn

from skorch.utils import to_tensor

from .base import BaseDANetwork
from .utils import check_generator


class BaseAdversarial(BaseDANetwork):
    """Domain-Adversarial Training of Neural Networks (DANN).

    From [1]_.

    Parameters
    ----------
    module : torch module (class or instance)
        A PyTorch :class:`~torch.nn.Module`. In general, the
        uninstantiated class should be passed, although instantiated
        modules will also work.
    criterion : torch criterion (class)
        The uninitialized criterion (loss) used to optimize the
        module.
    layer_names : list of tuples
        The names of the module's layers whose outputs are
        collected during the training.
    domain_classifier : torch module (class or instance)
        A PyTorch :class:`~torch.nn.Module` used for classying domain.
        In general, the uninstantiated class should be passed, although
        instantiated modules will also work.
    domain_criterion : torch criterion (class)
        The uninitialized criterion (loss) used to optimize the
        domain classifier.
    reg: float, optional (default=1)
        The regularization parameter of the domain classifier.

    References
    ----------
    .. [1]  Yaroslav Ganin et. al. Domain-Adversarial Training
            of Neural Networks  In Journal of Machine Learning
            Research, 2016.
    """

    def __init__(
        self,
        module,
        criterion,
        layer_names,
        domain_classifier,
        domain_criterion=torch.nn.BCELoss,
        **kwargs
    ):
        super().__init__(
            module, criterion, layer_names, **kwargs
        )
        self.domain_classifier = domain_classifier
        self.domain_criterion = domain_criterion

    def initialize_criterion(self):
        """Initializes the criterion.

        If the criterion is already initialized and no parameter was changed, it
        will be left as is.

        """
        kwargs = self.get_params_for('domain_criterion')
        domain_criterion = self.initialized_instance(self.domain_criterion, kwargs)
        self.domain_criterion_ = domain_criterion
        return super().initialize_criterion()

    def initialize_module(self):
        """Initializes the module and add hooks to return features.

        If the module is already initialized and no parameter was changed, it
        will be left as is.
        """
        kwargs = self.get_params_for('domain_classifier')
        domain_classifier = self.initialized_instance(self.domain_classifier, kwargs)
        self.domain_classifier_ = domain_classifier

        return super().initialize_module()


class DANN(BaseAdversarial):
    """Domain-Adversarial Training of Neural Networks (DANN).

    From [1]_.

    Parameters
    ----------
    module : torch module (class or instance)
        A PyTorch :class:`~torch.nn.Module`. In general, the
        uninstantiated class should be passed, although instantiated
        modules will also work.
    criterion : torch criterion (class)
        The uninitialized criterion (loss) used to optimize the
        module.
    layer_names : list of tuples
        The names of the module's layers whose outputs are
        collected during the training.
    domain_classifier : torch module (class or instance)
        A PyTorch :class:`~torch.nn.Module` used for classying domain.
        In general, the uninstantiated class should be passed, although
        instantiated modules will also work.
    domain_criterion : torch criterion (class)
        The uninitialized criterion (loss) used to optimize the
        domain classifier.
    reg: float, optional (default=1)
        The regularization parameter of the domain classifier.

    References
    ----------
    .. [1]  Yaroslav Ganin et. al. Domain-Adversarial Training
            of Neural Networks  In Journal of Machine Learning
            Research, 2016.
    """

    def __init__(
        self,
        module,
        criterion,
        layer_names,
        domain_classifier,
        domain_criterion=torch.nn.BCELoss,
        reg=1,
        **kwargs
    ):
        super().__init__(
            module,
            criterion,
            layer_names,
            domain_classifier,
            domain_criterion,
            **kwargs
        )
        self.reg = reg

    def _get_loss_da(
        self,
        y_pred,
        y_true,
        embedd,
        embedd_target,
        X=None,
        X_target=None,
        y_pred_target=None,
        training=True
    ):
        """Compute the domain adaptation loss"""
        y_true = to_tensor(y_true, device=self.device)
        loss_DANN = 0
        dtype = torch.float32
        for i in range(len(embedd)):
            # create domain label
            domain_label = torch.zeros(
                (embedd[i].size()[0]), device=self.device, dtype=dtype
            )
            domain_label_target = torch.ones(
                (embedd_target[i].size()[0]), device=self.device, dtype=dtype
            )

            # update classification function
            output_domain = self.domain_classifier_.forward(
                embedd[i], self.reg
            ).flatten()
            output_domain_target = self.domain_classifier_.forward(
                embedd_target[i], self.reg
            ).flatten()
            loss_DANN += (
                self.domain_criterion_(output_domain, domain_label) +
                self.domain_criterion_(output_domain_target, domain_label_target)
            )
        loss_classif = self.criterion_(y_pred, y_true)

        return loss_classif + loss_DANN, loss_classif, loss_DANN


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


class CDAN(BaseAdversarial):
    """Conditional Domain Adversarial Networks (CDAN).

    From [1]_.

    Parameters
    ----------
    module : torch module (class or instance)
        A PyTorch :class:`~torch.nn.Module`. In general, the
        uninstantiated class should be passed, although instantiated
        modules will also work.
    criterion : torch criterion (class)
        The uninitialized criterion (loss) used to optimize the
        module.
    layer_names : list of tuples
        The names of the module's layers whose outputs are
        collected during the training.
    domain_classifier : torch module (class or instance)
        A PyTorch :class:`~torch.nn.Module` used for classifying domain.
        The input size of the domain classifier should be
        min(max_features, embedding_size \times n_classes).
        In general, the uninstantiated class should be passed, although
        instantiated modules will also work.
    domain_criterion : torch criterion (class), default=BCELoss
        The uninitialized criterion (loss) used to optimize the
        domain classifier.
    reg: float, (default=1)
        The regularization parameter of the domain classifier.
    max_features : int, default=4096
        Maximum size of the input for the domain classifier.
        4096 is the largest number of units in typical deep network
        according to [1]_.
    random_state : int, Generator instance or None, default=None
        Determines random number generation for random layer creation. Pass an int
        for reproducible output across multiple function calls.

    References
    ----------
    .. [1]  Mingsheng Long et. al. Conditional Adversarial Domain Adaptation
            In NeurIPS, 2016.
    """

    def __init__(
        self,
        module,
        criterion,
        layer_names,
        domain_classifier,
        domain_criterion=torch.nn.BCELoss,
        reg=1,
        max_features=4096,
        random_state=None,
        **kwargs
    ):
        super().__init__(
            module,
            criterion,
            layer_names,
            domain_classifier,
            domain_criterion,
            **kwargs
        )
        self.reg = reg
        self.max_features = max_features
        self.random_state = random_state

    def _get_loss_da(
        self,
        y_pred,
        y_true,
        embedd,
        embedd_target,
        X=None,
        X_target=None,
        y_pred_target=None,
        training=True
    ):
        """Compute the domain adaptation loss"""
        y_true = to_tensor(y_true, device=self.device)
        loss_CDAN = 0
        dtype = torch.float32
        n_classes = y_pred.shape[1]
        gen = check_generator(self.random_state)
        for i in range(len(embedd)):
            n_features = embedd[i].shape[1]
            if n_features * n_classes > self.max_features:
                random_layer = _RandomLayer(
                    gen,
                    input_dims=[n_features, n_classes],
                    output_dim=self.max_features
                )
            else:
                random_layer = None

            # Compute the input for the domain classifier
            if random_layer is None:
                multilinear_map = torch.bmm(
                    y_pred.unsqueeze(2), embedd[i].unsqueeze(1))
                multilinear_map_target = torch.bmm(
                    y_pred_target.unsqueeze(2), embedd_target[i].unsqueeze(1))

                multilinear_map = multilinear_map.view(-1, n_features * n_classes)
                multilinear_map_target = multilinear_map_target.view(
                        -1, n_features * n_classes)

            else:
                multilinear_map = random_layer.forward([embedd[i], y_pred])
                multilinear_map_target = random_layer.forward(
                    [embedd_target[i], y_pred_target])

            # Compute the output of the domain classifier
            output_domain = self.domain_classifier_.forward(
               multilinear_map, self.reg
            ).flatten()
            output_domain_target = self.domain_classifier_.forward(
                multilinear_map_target, self.reg
            ).flatten()

            # create domain label
            domain_label = torch.zeros(
                (embedd[i].size()[0]), device=self.device, dtype=dtype
            )
            domain_label_target = torch.ones(
                (embedd_target[i].size()[0]), device=self.device, dtype=dtype
            )

            # update classification function
            loss_CDAN += (
                self.domain_criterion_(output_domain, domain_label) +
                self.domain_criterion_(output_domain_target, domain_label_target)
            )

        loss_classif = self.criterion_(y_pred, y_true)
        return loss_classif + loss_CDAN, loss_classif, loss_CDAN
