import torch
from torch import nn
from torch.autograd import Function

import numpy as np

import pytest

from skada.feature import DANN
from skada.utils import NeuralNetwork


class ReverseLayerF(Function):
    """XXX add docstring"""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

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
        lamb=0.1,
        dropout=0.25,
        n_classes=2
    ):
        super().__init__()
        self.lamb = lamb
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(len_last_layer, n_classes)
        )

    def forward(self, x):
        """Forward pass.
        Parameters
        ---------
        x: torch.Tensor
            Batch of EEG windows of shape (batch_size, n_channels, n_times).
        lamb: float
            Parameter for the reverse layer
        """
        reverse_x = ReverseLayerF.apply(x, self.lamb)

        return self.classifier(reverse_x)


@pytest.mark.parametrize(
    "input_size, n_channels, n_classes",
    [(100, 2, 5), (120, 1, 3)],
)
def test_dann(input_size, n_channels, n_classes):
    rng = np.random.RandomState(42)
    n_examples = 20

    module = NeuralNetwork(
        n_channels=n_channels, input_size=input_size, n_classes=n_classes, kernel_size=8
    )
    module.eval()

    rng = np.random.RandomState(42)
    X = rng.randn(n_examples, n_channels, input_size)
    X = torch.from_numpy(X.astype(np.float32))
    y = rng.randint(n_classes, size=n_examples)
    y = torch.from_numpy(y)
    X_target = rng.randn(n_examples, n_channels, input_size)
    X_target = torch.from_numpy(X_target.astype(np.float32))
    y_target = rng.randint(n_classes, size=n_examples)
    y_target = torch.from_numpy(y_target)

    method = DANN(
        module=module,
        criterion=nn.CrossEntropyLoss(),
        layer_names=["feature_extractor"],
        max_epochs=2,
        domain_classifier=DomainClassifier,
        domain_classifier__len_last_layer=module.len_last_layer,
        domain_criterion=nn.CrossEntropyLoss,
    )
    method.fit(X, y, X_target=X_target)
    y_pred = method.predict(X_target)

    assert y_pred.shape[0] == n_examples
