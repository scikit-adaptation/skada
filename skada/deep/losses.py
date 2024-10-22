# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Yanis Lalou <yanis.lalou@polytechnique.edu>
#         Antoine Collas <contact@antoinecollas.fr>
#
# License: BSD 3-Clause

import warnings
from functools import partial

import ot
import skorch  # noqa: F401
import torch  # noqa: F401
import torch.nn.functional as F
from torch.nn.functional import mse_loss

from skada.deep.base import BaseDALoss
from skada.deep.utils import SphericalKMeans


def deepcoral_loss(features, features_target, assume_centered=False):
    """Estimate the Frobenius norm divide by 4*n**2
       for DeepCORAL method [12]_.

    Parameters
    ----------
    features : tensor
        Source features.
    features_target : tensor
        Target features.
    assume_centered: bool, default=False
        If True, data are not centered before computation.

    Returns
    -------
    loss : ndarray
        The loss of the method.

    References
    ----------
    .. [12] Baochen Sun and Kate Saenko. Deep coral:
            Correlation alignment for deep domain
            adaptation. In ECCV Workshops, 2016.
    """
    if not assume_centered:
        features = features - features.mean(0)
        features_target = features_target - features_target.mean(0)
    cov = torch.cov(features.T)
    cov_target = torch.cov(features_target.T)
    divergence = mse_loss(cov, cov_target, reduction="sum")
    dim = features.shape[1]
    loss = (1 / (4 * (dim**2))) * divergence
    return loss


def deepjdot_loss(
    y_s,
    y_pred_t,
    features_s,
    features_t,
    reg_dist,
    reg_cl,
    sample_weights=None,
    target_sample_weights=None,
    criterion=None,
):
    """Compute the OT loss for DeepJDOT method [13]_.

    Parameters
    ----------
    y_s : tensor
        labels of the source data used to perform the distance matrix.
    y_pred_t : tensor
        labels of the target data used to perform the distance matrix.
    features_s : tensor
        features of the source data used to perform the distance matrix.
    features_t : tensor
        features of the target data used to perform the distance matrix.
    reg_dist : float
        Divergence term regularization parameter.
    reg_cl : float, default=1
        Class distance term regularization parameter.
    sample_weights : tensor
        Weights of the source samples.
        If None, create uniform weights.
    target_sample_weights : tensor
        Weights of the source samples.
        If None, create uniform weights.
    criterion : torch criterion (class)
        The criterion (loss) used to compute the
        DeepJDOT loss. If None, use the CrossEntropyLoss.

    Returns
    -------
    loss : ndarray
        The loss of the method.

    References
    ----------
    .. [13]  Bharath Bhushan Damodaran, Benjamin Kellenberger,
            Remi Flamary, Devis Tuia, and Nicolas Courty.
            DeepJDOT: Deep Joint Distribution Optimal Transport
            for Unsupervised Domain Adaptation. In ECCV 2018
            15th European Conference on Computer Vision,
            September 2018. Springer.
    """
    dist = torch.cdist(features_s, features_t, p=2) ** 2

    y_target_matrix = y_pred_t.repeat(len(y_pred_t), 1, 1).permute(1, 2, 0)

    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss(reduction="none")

    loss_target = criterion(y_target_matrix, y_s.repeat(len(y_s), 1)).T
    M = reg_dist * dist + reg_cl * loss_target

    # Compute the loss
    if sample_weights is None:
        sample_weights = torch.full(
            (len(features_s),), 1.0 / len(features_s), device=features_s.device
        )
    if target_sample_weights is None:
        target_sample_weights = torch.full(
            (len(features_t),), 1.0 / len(features_t), device=features_t.device
        )
    loss = ot.emd2(sample_weights, target_sample_weights, M)

    return loss


def _gaussian_kernel(x, y, sigmas):
    """Computes multi gaussian kernel between each pair of the two vectors."""
    sigmas = sigmas.view(sigmas.shape[0], 1)
    beta = 1.0 / sigmas
    dist = torch.cdist(x, y)
    dist_ = dist.view(1, -1)
    s = torch.matmul(beta, dist_)

    return torch.sum(torch.exp(-s), 0).view_as(dist)


def _maximum_mean_discrepancy(x, y, kernel):
    """Computes the maximum mean discrepancy between the vectors
    using the given kernel.
    """
    cost = torch.mean(kernel(x, x))
    cost += torch.mean(kernel(y, y))
    cost -= 2 * torch.mean(kernel(x, y))

    return cost


def dan_loss(features_s, features_t, sigmas=None):
    """Define the mmd loss based on multi-kernel defined in [14]_.

    Parameters
    ----------
    features_s : tensor
        Source features used to compute the mmd loss.
    features_t : tensor
        Target features used to compute the mmd loss.
    sigmas : array like, default=None,
        If array, sigmas used for the multi gaussian kernel.
        If None, uses sigmas proposed  in [1]_.

    Returns
    -------
    loss : float
        The loss of the method.

    References
    ----------
    .. [14]  Mingsheng Long et. al. Learning Transferable
            Features with Deep Adaptation Networks.
            In ICML, 2015.
    """
    if sigmas is None:
        median_pairwise_distance = torch.median(torch.cdist(features_s, features_s))
        sigmas = (
            torch.tensor([2 ** (-8) * 2 ** (i * 1 / 2) for i in range(33)]).to(
                features_s.device
            )
            * median_pairwise_distance
        )
    else:
        sigmas = torch.tensor(sigmas).to(features_s.device)

    gaussian_kernel = partial(_gaussian_kernel, sigmas=sigmas)

    loss = _maximum_mean_discrepancy(features_s, features_t, kernel=gaussian_kernel)

    return loss


def cdd_loss(
    y_s,
    features_s,
    features_t,
    target_kmeans=None,
    sigmas=None,
    distance_threshold=0.5,
    class_threshold=3,
):
    """Define the contrastive domain discrepancy loss based on [33]_.

    Parameters
    ----------
    y_s : tensor
        labels of the source data used to compute the loss.
    features_s : tensor
        features of the source data used to compute the loss.
    features_t : tensor
        features of the target data used to compute the loss.
    target_kmeans : SphericalKMeans
        Pre-computed target KMeans clustering model.
    sigmas : array like, default=None,
        If array, sigmas used for the multi gaussian kernel.
        If None, uses sigmas proposed  in [1]_.
    distance_threshold : float, optional (default=0.5)
        Distance threshold for discarding the samples that are
        to far from the centroids.
    class_threshold : int, optional (default=3)
        Minimum number of samples in a class to be considered for the loss.

    Returns
    -------
    loss : float
        The loss of the method.

    References
    ----------
    .. [33] Kang, G., Jiang, L., Yang, Y., & Hauptmann, A. G. (2019).
           Contrastive adaptation network for unsupervised domain adaptation.
           In Proceedings of the IEEE/CVF Conference on Computer Vision
           and Pattern Recognition (pp. 4893-4902).
    """
    n_classes = len(y_s.unique())

    # Use pre-computed cluster_labels_t
    if target_kmeans is None:
        warnings.warn(
            "Source centroids are not computed for the whole training set, "
            "computing them on the current batch set."
        )

        source_centroids = []

        for c in range(n_classes):
            mask = y_s == c
            if mask.sum() > 0:
                class_features = features_s[mask]
                normalized_features = F.normalize(class_features, p=2, dim=1)
                centroid = normalized_features.sum(dim=0)
                source_centroids.append(centroid)

        # Use source centroids to initialize target clustering
        target_kmeans = SphericalKMeans(
            n_clusters=n_classes,
            random_state=0,
            centroids=source_centroids,
            device=features_t.device,
        )
        target_kmeans.fit(features_t)

    # Predict clusters for target samples
    cluster_labels_t = target_kmeans.predict(features_t)

    # Discard ambiguous target samples
    similarities = F.cosine_similarity(
        features_t.unsqueeze(1), target_kmeans.cluster_centers_.unsqueeze(0)
    )
    mask_t = 0.5 * (1 - similarities.max(dim=1)[0]) < distance_threshold
    features_t = features_t[mask_t]
    cluster_labels_t = cluster_labels_t[mask_t]

    # Discard ambiguous classes
    class_counts = torch.bincount(cluster_labels_t, minlength=n_classes)
    valid_classes = class_counts >= class_threshold
    mask_t = valid_classes[cluster_labels_t]
    features_t = features_t[mask_t]
    cluster_labels_t = cluster_labels_t[mask_t]

    # Define sigmas
    if sigmas is None:
        median_pairwise_distance = torch.median(torch.cdist(features_s, features_s))
        sigmas = (
            torch.tensor([2 ** (-8) * 2 ** (i * 1 / 2) for i in range(33)]).to(
                features_s.device
            )
            * median_pairwise_distance
        )
    else:
        sigmas = torch.tensor(sigmas).to(features_s.device)

    # Compute CDD
    intraclass = 0
    interclass = 0

    for c1 in range(n_classes):
        for c2 in range(c1, n_classes):
            if valid_classes[c1] and valid_classes[c2]:
                # Compute e1
                kernel_ss = _gaussian_kernel(features_s, features_s, sigmas)
                mask_c1_c1 = (y_s == c1).float()
                e1 = (kernel_ss * mask_c1_c1).sum() / (mask_c1_c1.sum() ** 2)

                # Compute e2
                kernel_tt = _gaussian_kernel(features_t, features_t, sigmas)
                mask_c2_c2 = (cluster_labels_t == c2).float()
                e2 = (kernel_tt * mask_c2_c2).sum() / (mask_c2_c2.sum() ** 2)

                # Compute e3
                kernel_st = _gaussian_kernel(features_s, features_t, sigmas)
                mask_c1 = (y_s == c1).float().unsqueeze(1)
                mask_c2 = (cluster_labels_t == c2).float().unsqueeze(0)
                mask_c1_c2 = mask_c1 * mask_c2
                e3 = (kernel_st * mask_c1_c2).sum() / (mask_c1_c2.sum() ** 2)

                if c1 == c2:
                    intraclass += e1 + e2 - 2 * e3
                else:
                    interclass += e1 + e2 - 2 * e3

    cdd = (intraclass / len(valid_classes)) - (
        interclass / (len(valid_classes) ** 2 - len(valid_classes))
    )

    return cdd


class TestLoss(BaseDALoss):
    """Test Loss to check the deep API"""

    def __init__(
        self,
    ):
        super().__init__()

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
        return 0
