import numpy as np
import torch
import pytest

from skada.features import DeepCORAL
from skada.utils import NeuralNetwork, CustomDataset


@pytest.mark.parametrize(
    "input_size,n_classes",
    [(100, 5), (80, 4), (120, 2)],
)
def test_deepcoral(input_size, n_classes):
    rng = np.random.RandomState(42)
    n_examples = 20

    model = NeuralNetwork(
        input_size=input_size, n_classes=n_classes
    )
    model.eval()

    rng = np.random.RandomState(42)
    X = rng.randn(n_examples, input_size)
    X = torch.from_numpy(X.astype(np.float32))
    y = rng.randint(n_classes, size=n_examples)
    y = torch.from_numpy(y)
    X_target = rng.randn(n_examples, input_size)
    X_target = torch.from_numpy(X_target.astype(np.float32))
    y_target = rng.randint(n_classes, size=n_examples)
    y_target = torch.from_numpy(y_target)

    dataset = CustomDataset(X, y)
    dataset_target = CustomDataset(X_target, y_target)

    method = DeepCORAL(
        base_model=model,
        layer_names=["feature_extractor"],
        batch_size=8,
        n_epochs=2
    )
    method.fit(dataset=dataset, dataset_target=dataset_target)
    y_pred = method.predict(dataset_target)

    assert y_pred.shape[0] == n_examples
