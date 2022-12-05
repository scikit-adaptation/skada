<<<<<<< HEAD
=======
import torch
from torch import nn
from torch.utils.data import Dataset

import ot

from functools import partial


def cov(x, eps=1e-5):
    """Estimate the covariance matrix"""
    assert len(x.size()) == 2, x.size()

    N, d = x.size()
    reg = torch.diag(torch.full((d,), eps)).to(x.device)
    x_ = x - x.mean(dim=0, keepdim=True)

    return torch.einsum("ni,nj->ij", (x_, x_)) / (N - 1) + reg


def norm_coral(A, B):
    """Estimate the Frobenius norm divide by 4*n**2"""
    diff = A - B
    return (diff * diff).sum() / (4 * len(A) ** 2)


def get_intermediate_layers(intermediate_layers, layer_name):
    def hook(model, input, output):
        intermediate_layers[layer_name] = output.flatten(start_dim=1)

    return hook


def register_forwards_hook(module, intermediate_layers, layer_names):
    for layer_name, layer_module in module.named_modules():
        if layer_name in layer_names:
            layer_module.register_forward_hook(
                get_intermediate_layers(intermediate_layers, layer_name)
            )


def ot_solve(a, b, M, num_iter_max=100000):
    a2 = a.detach().cpu().numpy()
    b2 = b.detach().cpu().numpy()
    M2 = M.detach().cpu().numpy()

    # project on simplex for float64 or else numerical errors
    a2 /= a2.sum()
    b2 /= b2.sum()

    G = ot.emd(a2, b2, M2, log=False, numItermax=num_iter_max)
    return torch.from_numpy(G).to(a.device)


def jdot_distance_matrix(
    embedd,
    embedd_target,
    y,
    y_target,
    reg_d,
    reg_cl,
    class_weights=None,
    n_classes=3
):
    """Compute the distance matrix for DeepJDOT method

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
    class_weight : array, shape=(n_classes)
        Weight of classes to compute target classes loss.
        If None, don't use weights.
    n_classes : int, default=2
        Number of classes in the data.
    """
    if class_weights is None:
        weights = torch.ones(n_classes).to(embedd.device)
    else:
        weights = torch.Tensor(class_weights).to(embedd.device)

    dist = torch.cdist(embedd, embedd_target, p=2) ** 2

    onehot_y_source = torch.nn.functional.one_hot(y, num_classes=n_classes).to(
        device=y.device, dtype=embedd.dtype
    )
    loss_target = (weights @ onehot_y_source.T).reshape(len(y), 1) * (
        -(onehot_y_source @ y_target.T) + torch.logsumexp(y_target, dim=1)
    )
    M = reg_d * dist + reg_cl * loss_target

    return M


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
>>>>>>> refs/rewritten/main-4
