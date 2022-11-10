import torch

import ot

from torch import nn
from torch.utils.data import Dataset


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
        intermediate_layers[layer_name] = output.flatten(start_dim=1).detach()

    return hook


def register_forwards_hook(module, intermediate_layers, layer_names):
    for layer_name, layer_module in module.named_modules():
        if layer_name in layer_names:
            layer_module.register_forward_hook(
                get_intermediate_layers(intermediate_layers, layer_name)
            )


def ot_solve(a, b, M, num_iter_max=100000):
    a2 = a.detach().cpu().numpy()  # .astype(np.float64)
    b2 = b.detach().cpu().numpy()  # .astype(np.float64)
    M2 = M.detach().cpu().numpy()  # .astype(np.float64)

    # project on simplex for float64 or else numerical errors
    a2 /= a2.sum()
    b2 /= b2.sum()

    G = ot.emd(a2, b2, M2, log=False, numItermax=num_iter_max)
    return torch.from_numpy(G).to(a.device)


def distance_matrix(embedd_source, embedd_target, y_source, y_target, alpha, beta, class_weights, n_classes=3):
    if class_weights is None:
        weights = torch.ones(n_classes)
    else:
        weights = torch.Tensor(class_weights).to(embedd_source.device)

    dist = torch.cdist(embedd_source, embedd_target, p=2) ** 2

    onehot_y_source = torch.nn.functional.one_hot(y_source, num_classes=n_classes).to(
        device=y_source.device, dtype=embedd_source.dtype
    )
    loss_target = (weights @ onehot_y_source.T).reshape(len(y_source), 1) * (
        -(onehot_y_source @ y_target.T) + torch.logsumexp(y_target, dim=1)
    )
    M = alpha * dist + beta * loss_target

    return M


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, n_classes):
        super(NeuralNetwork, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 10),
            nn.ReLU(),
        )
        self.fc = nn.Linear(10, n_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fc(x.flatten(start_dim=1))
        return x


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
