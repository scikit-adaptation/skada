import torch
from torch import nn

import numpy as np

import pytest

from skada.feature import DeepCORAL
from skada.utils import NeuralNetwork


@pytest.mark.parametrize(
    "input_size, n_channels, n_classes",
    [(100, 2, 5), (120, 1, 3)],
)
def test_deepcoral(input_size, n_channels, n_classes):
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

    method = DeepCORAL(
        module=module,
        criterion=nn.CrossEntropyLoss(),
        layer_names=["feature_extractor"],
        max_epochs=2
    )
    method.fit(X, y, X_target=X_target)
    y_pred = method.predict(X_target)

    assert y_pred.shape[0] == n_examples
