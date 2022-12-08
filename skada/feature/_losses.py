# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD 3-Clause

import torch

import ot

from functools import partial


def deepcoral_loss(cov, cov_target):
    """Estimate the Frobenius norm divide by 4*n**2
       for DeepCORAL method [1]_.

    Parameters
    ----------
    cov : tensor
        Covariance of the embeddings of the source data.
    cov_target : tensor
        Covariance of the embeddings of the target data.

    Returns
    -------
    loss : ndarray
        The loss of the method.

    References
    ----------
    .. [1]  Baochen Sun and Kate Saenko. Deep coral:
            Correlation alignment for deep domain
            adaptation. In ECCV Workshops, 2016.
    """
    diff = cov - cov_target
    loss = (diff * diff).sum() / (4 * len(cov) ** 2)
    return loss


def deepjdot_loss(
    embedd,
    embedd_target,
    y,
    y_target,
    reg_d,
    reg_cl,
    sample_weights=None,
    target_sample_weights=None,
    criterion=torch.nn.CrossEntropyLoss(),
):
    """Compute the OT loss for DeepJDOT method [1]_.

    Parameters
    ----------
    embedd : tensor
        embeddings of the source data used to perform the distance matrix.
    embedd_target : tensor
        embeddings of the target data used to perform the distance matrix.
    y : tensor
        labels of the source data used to perform the distance matrix.
    y_target : tensor
        labels of the target data used to perform the distance matrix.
    reg_d : float, default=1
        Distance term regularization parameter.
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
        DeepJDOT loss.

    Returns
    -------
    loss : ndarray
        The loss of the method.

    References
    ----------
    .. [1]  Bharath Bhushan Damodaran, Benjamin Kellenberger,
            Remi Flamary, Devis Tuia, and Nicolas Courty.
            DeepJDOT: Deep Joint Distribution Optimal Transport
            for Unsupervised Domain Adaptation. In ECCV 2018
            15th European Conference on Computer Vision,
            September 2018. Springer.
    """
    dist = torch.cdist(embedd, embedd_target, p=2) ** 2

    # y_matrix = torch.cat([y.unsqueeze(dim=0) for _ in range(len(y))], axis=0)
    # y_target_matrix = torch.cat([y_target.unsqueeze(dim=2)
    #                             for _ in range(len(y_target))], axis=2)
    y_target_matrix = y_target.repeat(len(y_target), 1, 1).permute(1, 2, 0)

    loss_target = criterion(y_target_matrix, y.repeat(len(y), 1)).T

    M = reg_d * dist + reg_cl * loss_target

    # Compute the loss
    if sample_weights is None:
        sample_weights = torch.full(
            (len(embedd),),
            1.0 / len(embedd),
            device=embedd.device
        )
    if target_sample_weights is None:
        target_sample_weights = torch.full(
            (len(embedd_target),),
            1.0 / len(embedd_target),
            device=embedd_target.device
        )
    loss = ot.emd2(sample_weights, target_sample_weights, M)

    return loss


def _gaussian_kernel(x, y, sigmas):
    """Computes multi gaussian kernel between each pair of the two vectors."""
    sigmas = sigmas.view(sigmas.shape[0], 1)
    beta = 1. / sigmas
    dist = torch.cdist(x, y)
    dist_ = dist.view(1, -1)
    s = torch.matmul(beta, dist_)

    return torch.sum(torch.exp(-s), 0).view_as(dist)


def _maximum_mean_discrepancy(x, y, kernel):
    """Computes the maximum mean discrepency between the vectors
       using the given kernel."""
    cost = torch.mean(kernel(x, x))
    cost += torch.mean(kernel(y, y))
    cost -= 2 * torch.mean(kernel(x, y))

    return cost


def dan_loss(source_features, target_features, sigmas=None):
    """Define the mmd loss based on multi-kernel defined in [1]_.

    Parameters
    ----------
    source_features : tensor
        Source features used to compute the mmd loss.
    target_features : tensor
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
    .. [1]  Mingsheng Long et. al. Learning Transferable
            Features with Deep Adaptation Networks.
            In ICML, 2015.
    """
    if sigmas is None:
        median_pairwise_distance = torch.median(
            torch.cdist(source_features, source_features)
        )
        sigmas = torch.tensor(
            [2**(-8) * 2**(i*1/2) for i in range(33)]
        ).to(source_features.device) * median_pairwise_distance
    else:
        sigmas = torch.tensor(sigmas).to(source_features.device)

    gaussian_kernel = partial(
        _gaussian_kernel, sigmas=sigmas
    )

    loss = _maximum_mean_discrepancy(
        source_features, target_features, kernel=gaussian_kernel
    )

    return loss
