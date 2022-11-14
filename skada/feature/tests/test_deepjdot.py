import torch
from torch import nn

import numpy as np

import pytest

from skada.feature import DeepJDOT
from skada.utils import NeuralNetwork


@pytest.mark.parametrize(
    "input_size,n_classes",
    [(100, 5), (120, 3)],
)
def test_deepjdot(input_size, n_classes):
    rng = np.random.RandomState(42)
    n_examples = 20

    module = NeuralNetwork(
        input_size=input_size, n_classes=n_classes
    )
    module.eval()

    rng = np.random.RandomState(42)
    X = rng.randn(n_examples, input_size)
    X = torch.from_numpy(X.astype(np.float32))
    y = rng.randint(n_classes, size=n_examples)
    y = torch.from_numpy(y)
    X_target = rng.randn(n_examples, input_size)
    X_target = torch.from_numpy(X_target.astype(np.float32))
    y_target = rng.randint(n_classes, size=n_examples)
    y_target = torch.from_numpy(y_target)

    method = DeepJDOT(
        module=module,
        criterion=nn.CrossEntropyLoss(),
        layer_names=["feature_extractor"],
        max_epochs=2,
        n_classes=n_classes
    )
    method.fit(X, y, X_target=X_target)
    y_pred = method.predict(X_target)

    assert y_pred.shape[0] == n_examples
