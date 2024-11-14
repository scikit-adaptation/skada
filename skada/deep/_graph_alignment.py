# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#
# License: BSD 3-Clause

import torch
from torch import nn

from skada.deep.base import (
    BaseDALoss,
    DomainAwareCriterion,
    DomainAwareModule,
    DomainAwareNet,
    DomainBalancedDataLoader,
)
from skada.deep.callbacks import BatchIndexCallback, ComputeMemoryBank
from skada.deep.losses import gda_loss

from .modules import DomainClassifier


class SPALoss(BaseDALoss):
    """Loss SPA.

    This loss tries to minimize the divergence between features with
    adversarial method. The weights are updated to make harder
    to classify domains (i.e., remove domain-specific features).

    See [15]_ for details.

    Parameters
    ----------
    target_criterion : torch criterion (class), default=None
        The initialized criterion (loss) used to compute the
        DANN loss. If None, a BCELoss is used.

    References
    ----------
    .. [15] Yaroslav Ganin et. al. Domain-Adversarial Training
            of Neural Networks  In Journal of Machine Learning
            Research, 2016.
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
        self.batch_indices = None

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
        loss_adv = self.domain_criterion_(
            domain_pred_s, domain_label
        ) + self.domain_criterion_(domain_pred_t, domain_label_target)

        loss_gda = self.reg_gsa * gda_loss(features_s, features_t)

        if self.memory_features is None:
            self.memory_features = torch.rand_like(features_t)

        if self.memory_outputs is None:
            self.memory_outputs = torch.rand_like(y_pred_t)

        dis = -torch.mm(features_t.detach(), self.memory_features.t())
        for di in range(dis.size(0)):
            print(self.batch_indices)
            dis[di, self.batch_indices[di]] = torch.max(dis)
        _, p1 = torch.sort(dis, dim=1)

        w = torch.zeros(features_t.size(0), self.memory_features.size(0)).to(
            features_t.device
        )
        for wi in range(w.size(0)):
            for wj in range(self.K):
                w[wi][p1[wi, wj]] = 1 / self.K
        weight_, pred = torch.max(w.mm(self.memory_outputs), 1)

        loss_ = nn.CrossEntropyLoss(reduction="none")(y_pred_t, pred)
        classifier_loss = torch.sum(weight_ * loss_) / (torch.sum(weight_).item())
        loss_pl = self.reg_nap * classifier_loss

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
            of Neural Networks  In Journal of Machine Learning
            Research, 2016.
    """
    if domain_classifier is None:
        # raise error if num_feature is None
        if num_features is None:
            raise ValueError(
                "If domain_classifier is None, num_features has to be provided"
            )
        domain_classifier = DomainClassifier(num_features=num_features)

    if callbacks is None:
        callbacks = [ComputeMemoryBank(), BatchIndexCallback()]
    else:
        if isinstance(callbacks, list):
            callbacks.append(ComputeMemoryBank())
            callbacks.append(BatchIndexCallback())
        else:
            callbacks = [callbacks, ComputeMemoryBank(), BatchIndexCallback()]
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
        ),
        callbacks=callbacks,
        **kwargs,
    )
    return net
