import torch

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


def register_forwards_hook(model, intermediate_layers, layer_names):
    for layer_name, layer_module in model.named_modules():
        if layer_name in layer_names:
            layer_module.register_forward_hook(
                get_intermediate_layers(intermediate_layers, layer_name)
            )


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, n_channels, n_classes):
        super(NeuralNetwork, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=n_channels, out_channels=8, kernel_size=10),
            nn.ReLU(),
        )
        self.fc = nn.Linear(self._len_last_layer(n_channels, input_size), n_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fc(x.flatten(start_dim=1))
        return x

    def _len_last_layer(self, n_channels, input_size):
        self.feature_extractor.eval()
        with torch.no_grad():
            out = self.feature_extractor(torch.Tensor(1, n_channels, input_size))
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
