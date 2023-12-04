# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#
# License: BSD 3-Clause

import torch
from torch import nn
from torch.autograd import Function
from torch.utils.data import Dataset

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
    criterion=None,
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
        DeepJDOT loss. If None, use the CrossEntropyLoss.

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

    y_target_matrix = y_target.repeat(len(y_target), 1, 1).permute(1, 2, 0)

    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()

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


class NeuralNetwork(nn.Module):
    def __init__(
        self, n_channels, input_size, n_classes, kernel_size=64, out_channels=10
    ):
        super(NeuralNetwork, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(n_channels, out_channels, kernel_size),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size)
        )
        self.len_last_layer = self._len_last_layer(n_channels, input_size)
        self.fc = nn.Linear(self.len_last_layer, n_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fc(x.flatten(start_dim=1))
        return x

    def _len_last_layer(self, n_channels, input_size):
        self.feature_extractor.eval()
        with torch.no_grad():
            out = self.feature_extractor(
                torch.Tensor(1, n_channels, input_size))
        self.feature_extractor.train()
        return len(out.flatten())


class CustomDataset(Dataset):
    def __init__(
        self,
        data,
        label=None,
    ):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = self.data[idx]
        if self.label is None:
            return X
        else:
            y = self.label[idx]
            return X, y


class MNISTtoUSPSNet(nn.Module):
    def __init__(self):
        super(MNISTtoUSPSNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return output


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, reg):
        ctx.reg = reg

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.reg

        return output, None


class DomainClassifier(nn.Module):
    """Classifier Architecture for DANN method.
    Parameters
    ----------
    n_channels : int
        Number of EEG channels.
    sfreq : float
        EEG sampling frequency.
    n_conv_chs : int
        Number of convolutional channels. Set to 8 in [1]_.
    time_conv_size_s : float
        Size of filters in temporal convolution layers, in seconds. Set to 0.5
        in [1]_ (64 samples at sfreq=128).
    max_pool_size_s : float
        Max pooling size, in seconds. Set to 0.125 in [1]_ (16 samples at
        sfreq=128).
    n_classes : int
        Number of classes.
    input_size_s : float
        Size of the input, in seconds.
    dropout : float
        Dropout rate before the output dense layer.
    """

    def __init__(
        self,
        len_last_layer,
        dropout=0.25,
        n_classes=1
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(len_last_layer, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, n_classes),
            nn.Sigmoid()
        )

    def forward(self, x, reg=None):
        """Forward pass.
        Parameters
        ---------
        x: torch.Tensor
            Batch of EEG windows of shape (batch_size, n_channels, n_times).
        lamb: float
            Parameter for the reverse layer
        """
        reverse_x = ReverseLayerF.apply(x, reg)
        return self.classifier(reverse_x)
