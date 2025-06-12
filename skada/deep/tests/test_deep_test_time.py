# Author : Maxence Barneche
#
# License: BSD-3-Clause

import pytest

torch = pytest.importorskip("torch")

import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader

from skada.datasets import make_shifted_datasets
from skada.deep._test_time import TestTimeCriterion, TestTimeNet
from skada.deep.base import DomainAwareModule
from skada.deep.losses import TestLoss
from skada.deep.modules import ToyModule2D


@pytest.mark.parametrize(
    "sd, epochs_adapt, optimizer_adapt",
    [(True, 3, Adam), (False, None, None)],
)
def test_test_time_criterion(sd, epochs_adapt, optimizer_adapt):
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
    sample_domain = sample_domain if sd else None

    method = TestTimeNet(
        DomainAwareModule(module, "dropout"),
        epochs_adapt=epochs_adapt,
        iterator_train=DataLoader,
        criterion=criterion,
        batch_size=10,
        max_epochs=2,
        train_split=None,
        optimizer_adapt=optimizer_adapt,
    )

    method.fit(X, y, sample_domain)

    # If the adaptation optimizer is specified, the net's optimizer should have changed
    # during the fit_adapt method called inside the fit method
    if optimizer_adapt is not None:
        assert isinstance(method.optimizer_, optimizer_adapt)
