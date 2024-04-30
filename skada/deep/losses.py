# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Yanis Lalou <yanis.lalou@polytechnique.edu>
#
# License: BSD 3-Clause

from functools import partial

import ot
import skorch  # noqa: F401
import torch  # noqa: F401

from skada.deep.base import BaseDALoss


def deepcoral_loss(features, features_target):
    """Estimate the Frobenius norm divide by 4*n**2
       for DeepCORAL method [12]_.

    Parameters
    ----------
    features : tensor
        Source features.
    features_target : tensor
        Target features.

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
    cov = torch.cov(features)
    cov_target = torch.cov(features_target)
    diff = cov - cov_target
    loss = (diff * diff).sum() / (4 * len(cov) ** 2)
    return loss


def deepjdot_loss(
    y_s,
    y_pred_t,
    features_s,
    features_t,
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
    M = dist + reg_cl * loss_target

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
