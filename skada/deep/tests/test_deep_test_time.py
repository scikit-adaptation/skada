# Author : Maxence Barneche
#
# License: BSD-3-Clause

import pytest

torch = pytest.importorskip("torch")

import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader

from skada.datasets import make_shifted_datasets
from skada.deep._test_time import Tent, TestTimeCriterion, TestTimeNet
from skada.deep.base import DomainAwareModule
from skada.deep.losses import TestLoss
from skada.deep.modules import ToyModule2D


@pytest.mark.parametrize(
    "epochs_adapt, optimizer_adapt, params_to_adapt, layers_to_adapt",
    [
        (None, None, None, None),  # Default test
        (3, None, None, None),  # Test with specific adaptation epochs given
        (None, Adam, None, None),
        (None, None, ["weight"], ["dropout"]),
    ],  # Test with Adam optimizer for adaptation
)
def test_test_time_net(epochs_adapt, optimizer_adapt, params_to_adapt, layers_to_adapt):
    num_features = 10
    module = ToyModule2D(num_features=num_features)
    criterion = TestTimeCriterion(torch.nn.CrossEntropyLoss(), TestLoss())

    n_samples = 20
    dataset = make_shifted_datasets(
        n_samples_source=n_samples,
        n_samples_target=n_samples,
        shift="conditional_shift",
        noise=0.1,
        random_state=42,
    )
    X, y, sample_domain = dataset
    X = X.astype(np.float32)

    method = TestTimeNet(
        DomainAwareModule(module, "dropout"),
        epochs_adapt=epochs_adapt,
        iterator_train=DataLoader,
        criterion=criterion,
        params_to_adapt=params_to_adapt,
        layers_to_adapt=layers_to_adapt,
        batch_size=10,
        max_epochs=2,
        train_split=None,
        optimizer_adapt=optimizer_adapt,
    )

    method.fit(X, y, sample_domain)
    method.fit_adapt(X, y, sample_domain)


def test_tent():
    num_features = 10
    module = ToyModule2D(num_features=num_features)

    n_samples = 20
    dataset = make_shifted_datasets(
        n_samples_source=n_samples,
        n_samples_target=n_samples,
        shift="conditional_shift",
        noise=0.1,
        random_state=42,
    )
    X, y, sample_domain = dataset
    X = X.astype(np.float32)

    method = Tent(
        DomainAwareModule(module, "dropout"),
        "dropout",
        epochs_adapt=3,
        iterator_train=DataLoader,
        batch_size=10,
        max_epochs=2,
        train_split=None,
    )

    method.fit(X, y, sample_domain)
    method.fit_adapt(X, y, sample_domain)
