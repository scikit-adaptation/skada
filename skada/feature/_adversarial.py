# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD 3-Clause

import torch
from skorch.utils import to_tensor

from .base import BaseDANetwork


class DANN(BaseDANetwork):
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
            module, criterion, layer_names, **kwargs
        )
        self.reg = reg
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
