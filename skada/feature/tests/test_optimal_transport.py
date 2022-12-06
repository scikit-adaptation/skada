import torch
from torch import nn

import pytest

from skada.feature import DeepJDOT
from skada.feature.architecture import toyCNN


@pytest.mark.parametrize(
    "input_size, n_channels, n_classes",
    [(100, 2, 5), (120, 1, 3)],
)
def test_deepjdot(input_size, n_channels, n_classes):
    module = toyCNN(
        n_channels=n_channels, input_size=input_size, n_classes=n_classes, kernel_size=8
    )
    module.eval()

    rng = torch.random.manual_seed(42)
    n_samples = 20
    X = torch.randn(size=(n_samples, n_channels, input_size), generator=rng)
    y = torch.randint(high=n_classes, size=(n_samples,), generator=rng)
    X_target = torch.randn(size=(n_samples, n_channels, input_size), generator=rng)

    batch_size = 16
    sample_weights = torch.full(
        (batch_size,),
        1.0 / batch_size
    )
    target_sample_weights = torch.full(
        (batch_size,),
        1.0 / batch_size
    )
    method = DeepJDOT(
        module=module,
        criterion=nn.CrossEntropyLoss(),
        layer_names=["feature_extractor"],
        max_epochs=2,
        n_classes=n_classes,
        batch_size=batch_size,
        sample_weights=sample_weights,
        target_sample_weights=target_sample_weights,
    )
    method.fit(X, y, X_target=X_target)
    y_pred = method.predict(X_target)

    assert y_pred.shape[0] == n_samples
