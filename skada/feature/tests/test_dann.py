import torch
from torch import nn
from torch.autograd import Function

import pytest

from skada.feature import DANN
from skada.utils import NeuralNetwork


class ReverseLayerF(Function):

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
    len_last_layer : int
        Size of the input, e.g size of the last layer of
        the feature extractor
    n_classes : int, default=1
        Number of classes
    """
    def __init__(
        self,
        len_last_layer,
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

    def forward(self, x, alpha=None):
        """Forward pass.

        Parameters
        ---------
        x: torch.Tensor
            Batch of EEG windows of shape (batch_size, n_channels, n_times).
        alpha: float
            Parameter for the reverse layer.
        """
        reverse_x = ReverseLayerF.apply(x, alpha)
        return self.classifier(reverse_x)


@pytest.mark.parametrize(
    "input_size, n_channels, n_classes",
    [(100, 2, 5), (120, 1, 3)],
)
def test_dann(input_size, n_channels, n_classes):
    module = NeuralNetwork(
        n_channels=n_channels, input_size=input_size, n_classes=n_classes, kernel_size=8
    )
    module.eval()

    rng = torch.random.manual_seed(42)
    n_samples = 20
    X = torch.randn(size=(n_samples, n_channels, input_size), generator=rng)
    y = torch.randint(high=n_classes, size=(n_samples,), generator=rng)
    X_target = torch.randn(size=(n_samples, n_channels, input_size), generator=rng)

    method = DANN(
        module=module,
        criterion=nn.CrossEntropyLoss(),
        layer_names=["feature_extractor"],
        max_epochs=2,
        domain_classifier=DomainClassifier,
        domain_classifier__len_last_layer=module.len_last_layer,
        domain_criterion=nn.BCELoss,
    )
    method.fit(X, y, X_target=X_target)
    y_pred = method.predict(X_target)

    assert y_pred.shape[0] == n_samples
