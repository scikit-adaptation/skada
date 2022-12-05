import torch
from torch import nn

import pytest

from skada.feature import DeepJDOT
from skada.feature.utils import NeuralNetwork


@pytest.mark.parametrize(
    "input_size, n_channels, n_classes",
    [(100, 2, 5), (120, 1, 3)],
)
def test_deepjdot(input_size, n_channels, n_classes):
    module = NeuralNetwork(
        n_channels=n_channels, input_size=input_size, n_classes=n_classes, kernel_size=8
    )
    module.eval()

    rng = torch.random.manual_seed(42)
    n_samples = 20
    X = torch.randn(size=(n_samples, n_channels, input_size), generator=rng)
    y = torch.randint(high=n_classes, size=(n_samples,), generator=rng)
    X_target = torch.randn(size=(n_samples, n_channels, input_size), generator=rng)

    method = DeepJDOT(
        module=module,
        criterion=nn.CrossEntropyLoss(),
        layer_names=["feature_extractor"],
        max_epochs=2,
        n_classes=n_classes
    )
    method.fit(X, y, X_target=X_target)
    y_pred = method.predict(X_target)

    assert y_pred.shape[0] == n_samples
