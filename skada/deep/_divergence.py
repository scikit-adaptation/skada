# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Yanis Lalou <yanis.lalou@polytechnique.edu>
#         Antoine Collas <contact@antoinecollas.fr>
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

from .callbacks import ComputeSourceCentroids
from .losses import cdd_loss, dan_loss, deepcoral_loss


class DeepCoralLoss(BaseDALoss):
    """Loss DeepCORAL

    This loss reduces the distance between covariances
    of the source features and the target features.
    See [12]_.

    Parameters
    ----------
    assume_centered: bool, default=False
        If True, data are not centered before computation.

    References
    ----------
    .. [12]  Baochen Sun and Kate Saenko. Deep coral:
            Correlation alignment for deep domain
            adaptation. In ECCV Workshops, 2016.
    """

    def __init__(
        self,
        assume_centered=False,
    ):
        super().__init__()
        self.assume_centered = assume_centered

    def forward(
        self,
        features_s,
        features_t,
        **kwargs,
    ):
        """Compute the domain adaptation loss"""
        loss = deepcoral_loss(features_s, features_t, self.assume_centered)
        return loss


def DeepCoral(
    module, layer_name, reg=1, assume_centered=False, base_criterion=None, **kwargs
):
    """DeepCORAL domain adaptation method.

    From [12]_.

    Parameters
    ----------
    module : torch module (class or instance)
        A PyTorch :class:`~torch.nn.Module`.
    layer_name : str
        The name of the module's layer whose outputs are
        collected during the training for the adaptation.
    reg : float, optional (default=1)
        Regularization parameter for DA loss.
    assume_centered: bool, default=False
        If True, data are not centered before computation.
    base_criterion : torch criterion (class)
        The base criterion used to compute the loss with source
        labels. If None, the default is `torch.nn.CrossEntropyLoss`.

    References
    ----------
    .. [12]  Baochen Sun and Kate Saenko. Deep coral:
            Correlation alignment for deep domain
            adaptation. In ECCV Workshops, 2016.
    """
    if base_criterion is None:
        base_criterion = torch.nn.CrossEntropyLoss()

    net = DomainAwareNet(
        module=DomainAwareModule,
        module__base_module=module,
        module__layer_name=layer_name,
        iterator_train=DomainBalancedDataLoader,
        criterion=DomainAwareCriterion,
        criterion__base_criterion=base_criterion,
        criterion__reg=reg,
        criterion__adapt_criterion=DeepCoralLoss(assume_centered=assume_centered),
        **kwargs,
    )
    return net


class DANLoss(BaseDALoss):
    """Loss DAN

    This loss reduces the MMD distance between
    source features and target features.
    From [14]_.

    Parameters
    ----------
    sigmas : array-like, optional (default=None)
        The sigmas for the Gaussian kernel.
    eps : float, default=1e-7
        Small constant added to median distance calculation for numerical stability.

    References
    ----------
    .. [14]  Mingsheng Long et. al. Learning Transferable
            Features with Deep Adaptation Networks.
            In ICML, 2015.
    """

    def __init__(self, sigmas=None, eps=1e-7):
        super().__init__()
        self.sigmas = sigmas
        self.eps = eps

    def forward(
        self,
        features_s,
        features_t,
        **kwargs,
    ):
        """Compute the domain adaptation loss"""
        loss = dan_loss(features_s, features_t, sigmas=self.sigmas, eps=self.eps)
        return loss


def DAN(module, layer_name, reg=1, sigmas=None, base_criterion=None, **kwargs):
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
        Regularization parameter for DA loss.
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
        iterator_train=DomainBalancedDataLoader,
        criterion=DomainAwareCriterion,
        criterion__base_criterion=base_criterion,
        criterion__reg=reg,
        criterion__adapt_criterion=DANLoss(sigmas=sigmas),
        **kwargs,
    )
    return net


class CANLoss(BaseDALoss):
    """Loss for Contrastive Adaptation Network (CAN)

    This loss implements the contrastive domain discrepancy (CDD)
    as described in [33].

    Parameters
    ----------
    distance_threshold : float, optional (default=0.5)
        Distance threshold for discarding the samples that are
        to far from the centroids.
    class_threshold : int, optional (default=3)
        Minimum number of samples in a class to be considered for the loss.
    sigmas : array like, default=None,
        If array, sigmas used for the multi gaussian kernel.
        If None, uses sigmas proposed  in [1]_.
    target_kmeans : sklearn KMeans instance, default=None,
        Pre-computed target KMeans clustering model.
    eps : float, default=1e-7
        Small constant added to median distance calculation for numerical stability.

    References
    ----------
    .. [33] Kang, G., Jiang, L., Yang, Y., & Hauptmann, A. G. (2019).
           Contrastive adaptation network for unsupervised domain adaptation.
           In Proceedings of the IEEE/CVF Conference on Computer Vision
           and Pattern Recognition (pp. 4893-4902).
    """

    def __init__(
        self,
        distance_threshold=0.5,
        class_threshold=3,
        sigmas=None,
        target_kmeans=None,
        eps=1e-7,
    ):
        super().__init__()
        self.distance_threshold = distance_threshold
        self.class_threshold = class_threshold
        self.sigmas = sigmas
        self.target_kmeans = target_kmeans
        self.eps = eps

    def forward(
        self,
        y_s,
        features_s,
        features_t,
        **kwargs,
    ):
        loss = cdd_loss(
            y_s,
            features_s,
            features_t,
            sigmas=self.sigmas,
            target_kmeans=self.target_kmeans,
            distance_threshold=self.distance_threshold,
            class_threshold=self.class_threshold,
            eps=self.eps,
        )

        return loss


def CAN(
    module,
    layer_name,
    reg=1,
    distance_threshold=0.5,
    class_threshold=3,
    sigmas=None,
    base_criterion=None,
    callbacks=None,
    **kwargs,
):
    """Contrastive Adaptation Network (CAN) domain adaptation method.

    From [33].

    Parameters
    ----------
    module : torch module (class or instance)
        A PyTorch :class:`~torch.nn.Module`.
    layer_name : str
        The name of the module's layer whose outputs are
        collected during the training for the adaptation.
    reg : float, optional (default=1)
        Regularization parameter for DA loss.
    distance_threshold : float, optional (default=0.5)
        Distance threshold for discarding the samples that are
        to far from the centroids.
    class_threshold : int, optional (default=3)
        Minimum number of samples in a class to be considered for the loss.
    sigmas : array like, default=None,
        If array, sigmas used for the multi gaussian kernel.
        If None, uses sigmas proposed  in [1]_.
    base_criterion : torch criterion (class)
        The base criterion used to compute the loss with source
        labels. If None, the default is `torch.nn.CrossEntropyLoss`.
    callbacks : list, optional
        List of callbacks to be used during training.

    References
    ----------
    .. [33] Kang, G., Jiang, L., Yang, Y., & Hauptmann, A. G. (2019).
           Contrastive adaptation network for unsupervised domain adaptation.
           In Proceedings of the IEEE/CVF Conference on Computer Vision
           and Pattern Recognition (pp. 4893-4902).
    """
    if base_criterion is None:
        base_criterion = torch.nn.CrossEntropyLoss()

    if callbacks is None:
        callbacks = [ComputeSourceCentroids()]
    else:
        if isinstance(callbacks, list):
            callbacks.append(ComputeSourceCentroids())
        else:
            callbacks = [callbacks, ComputeSourceCentroids()]

    net = DomainAwareNet(
        module=DomainAwareModule,
        module__base_module=module,
        module__layer_name=layer_name,
        iterator_train=DomainBalancedDataLoader,
        criterion=DomainAwareCriterion,
        criterion__base_criterion=base_criterion,
        criterion__reg=reg,
        criterion__adapt_criterion=CANLoss(
            distance_threshold=distance_threshold,
            class_threshold=class_threshold,
            sigmas=sigmas,
        ),
        callbacks=callbacks,
        **kwargs,
    )
    return net
