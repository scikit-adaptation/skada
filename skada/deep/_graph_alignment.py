# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#
# License: BSD 3-Clause

import torch

from skada.deep.base import (
    BaseDALoss,
    DomainAwareCriterion,
    DomainAwareModule,
    DomainAwareNet,
    DomainBalancedDataLoader,
)
from skada.deep.callbacks import ComputeMemoryBank
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
    target_criterion : torch criterion (class), default=None
        The initialized criterion (loss) used to compute the
        adversarial loss. If None, a BCELoss is used.
    reg_adv : float, default=1
        Regularization parameter for adversarial loss.
    reg_gsa : float, default=1
        Regularization parameter for graph alignment loss
    reg_nap : float, default=1
        Regularization parameter for nap loss
    num_samples : int, default=None
        Number of samples in the target domain.
    num_features : int, default=None
        Number of features in the network.
    num_classes : int, default=None
        Number of classes in the dataset.

    References
    ----------
    .. [36] Xiao et. al. SPA: A Graph Spectral Alignment Perspective for
            Domain Adaptation. In Neurips, 2023.
    """

    def __init__(
        self,
        domain_criterion=None,
        memory_features=None,
        memory_outputs=None,
        K=5,
        reg_adv=1,
        reg_gsa=1,
        reg_nap=1,
        num_samples=None,
        num_features=None,
        num_classes=None,
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
        if memory_features is None:
            if num_features is None or num_samples is None:
                raise ValueError(
                    "If memory_features is None, num_features"
                    "and num_samples has to be provided"
                )
            self.memory_features = torch.rand(num_samples, num_features)
        if memory_outputs is None:
            if num_classes is None or num_samples is None:
                raise ValueError(
                    "If memory_outputs is None, num_classes"
                    "and num_samples has to be provided"
                )
            self.memory_outputs = torch.rand(num_samples, num_classes)

    def forward(
        self,
        domain_pred_s,
        domain_pred_t,
        features_s,
        features_t,
        sample_idx_t,
        **kwargs,
    ):
        """Compute the domain adaptation loss"""
        self.sample_idx = sample_idx_t
        domain_label = torch.zeros(
            (domain_pred_s.size()[0]),
            device=domain_pred_s.device,
        )
        domain_label_target = torch.ones(
            (domain_pred_t.size()[0]),
            device=domain_pred_t.device,
        )

        # update classification function
        loss_adv = self.domain_criterion_(
            domain_pred_s, domain_label
        ) + self.domain_criterion_(domain_pred_t, domain_label_target)

        loss_gda = self.reg_gsa * gda_loss(features_s, features_t)

        loss_pl = self.reg_nap * nap_loss(
            features_s,
            features_t,
            self.memory_features,
            self.memory_outputs,
            K=self.K,
            sample_idx=self.sample_idx,
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
    num_samples_t=None,
    num_classes=None,
    base_criterion=None,
    domain_criterion=None,
    callbacks=None,
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
    num_samples_t : int, default=None
        Number of samples in the target domain.
    num_classes : int, default=None
        Number of classes in the dataset.
    base_criterion : torch criterion (class)
        The base criterion used to compute the loss with source
        labels. If None, the default is `torch.nn.CrossEntropyLoss`.
    domain_criterion : torch criterion (class)
        The criterion (loss) used to compute the
        DANN loss. If None, a BCELoss is used.

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
            ComputeMemoryBank(),
        ]
    else:
        if isinstance(callbacks, list):
            callbacks.append(ComputeMemoryBank())
        else:
            callbacks = [
                callbacks,
                ComputeMemoryBank(),
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
            num_samples=num_samples_t,
            num_features=num_features,
            num_classes=num_classes,
        ),
        callbacks=callbacks,
        **kwargs,
    )
    return net
