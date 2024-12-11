# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#
# License: BSD 3-Clause

import numpy as np
import torch

from skada.deep.base import (
    BaseDALoss,
    DomainAwareCriterion,
    DomainAwareModule,
    DomainAwareNet,
    DomainBalancedDataLoader,
)
from skada.deep.callbacks import CountEpochs, MemoryBank
from skada.deep.losses import gda_loss, nap_loss

from .modules import DomainClassifier


class SPALoss(BaseDALoss):
    """Loss SPA.

    This loss tries to minimize the divergence between features with
    adversarial method. The weights are updated to make harder
    to classify domains (i.e., remove domain-specific features).

    See [36]_ for details.

    Parameters
    ----------
    max_epochs : int
        Maximum number of epochs to train the model.
    target_criterion : torch criterion (class), default=None
        The initialized criterion (loss) used to compute the
        adversarial loss. If None, a BCELoss is used.
    reg_adv : float, default=1
        Regularization parameter for adversarial loss.
    reg_gsa : float, default=1
        Regularization parameter for graph alignment loss
    reg_nap : float, default=1
        Regularization parameter for nap loss

    References
    ----------
    .. [36] Xiao et. al. SPA: A Graph Spectral Alignment Perspective for
            Domain Adaptation. In Neurips, 2023.
    """

    def __init__(
        self,
        max_epochs,
        domain_criterion=None,
        memory_features=None,
        memory_outputs=None,
        K=5,
        reg_adv=1,
        reg_gsa=1,
        reg_nap=1,
    ):
        super().__init__()
        if domain_criterion is None:
            self.domain_criterion_ = torch.nn.BCELoss()
        else:
            self.domain_criterion_ = domain_criterion

        self.reg_adv = reg_adv
        self.reg_gsa = reg_gsa
        self.reg_nap = reg_nap
        self.K = K
        self.memory_features = memory_features
        self.memory_outputs = memory_outputs
        self.max_epochs = max_epochs
        self.n_epochs = 0

    def _scheduler_adv(self, high=1.0, low=0.0, alpha=10.0):
        max_epochs = self.max_epochs
        n_epochs = self.n_epochs
        return (
            2.0 * (high - low) / (1.0 + np.exp(-alpha * n_epochs / max_epochs))
            - (high - low)
            + low
        )

    def _scheduler_nap(self):
        return self.n_epochs / self.max_epochs

    def forward(
        self,
        y_pred_t,
        domain_pred_s,
        domain_pred_t,
        features_s,
        features_t,
        sample_idx_t,
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
        scale = self._scheduler_adv()
        loss_adv = (
            self.reg_adv
            * scale
            * (
                self.domain_criterion_(domain_pred_s, domain_label)
                + self.domain_criterion_(domain_pred_t, domain_label_target)
            )
        )

        loss_gda = self.reg_gsa * gda_loss(features_s, features_t, metric="gauss")

        scale = self._scheduler_nap()
        loss_pl = (
            self.reg_nap
            * scale
            * nap_loss(
                features_t=features_t,
                y_pred_t=y_pred_t,
                memory_features=self.memory_features,
                memory_outputs=self.memory_outputs,
                K=self.K,
                sample_idx_t=sample_idx_t,
            )
        )
        loss = loss_adv + loss_gda + loss_pl
        return loss


def SPA(
    module,
    layer_name,
    reg_adv=1,
    reg_gsa=1,
    reg_nap=1,
    domain_classifier=None,
    num_features=None,
    base_criterion=None,
    domain_criterion=None,
    callbacks=None,
    max_epochs=100,
    **kwargs,
):
    """Domain Adaptation with SPA.

    From [36]_.

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
        adversarial loss. If None, a BCELoss is used.
    callbacks : list, default=None
        List of callbacks to use during training.
    max_epochs : int, default=100
        Maximum number of epochs to train the model.

    References
    ----------
    .. [36] Xiao et. al. SPA: A Graph Spectral Alignment Perspective for
            Domain Adaptation. In Neurips, 2023.
    """
    if domain_classifier is None:
        # raise error if num_feature is None
        if num_features is None:
            raise ValueError(
                "If domain_classifier is None, num_features has to be provided"
            )
        domain_classifier = DomainClassifier(num_features=num_features)

    if callbacks is None:
        callbacks = [
            MemoryBank(),
            CountEpochs(),
        ]
    else:
        if isinstance(callbacks, list):
            callbacks.append(MemoryBank())
            callbacks.append(CountEpochs())
        else:
            callbacks = [
                callbacks,
                MemoryBank(),
                CountEpochs(),
            ]
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
        criterion__reg=1,
        criterion__adapt_criterion=SPALoss(
            domain_criterion=domain_criterion,
            reg_adv=reg_adv,
            reg_gsa=reg_gsa,
            reg_nap=reg_nap,
            max_epochs=max_epochs,
        ),
        callbacks=callbacks,
        max_epochs=max_epochs,
        **kwargs,
    )
    return net
