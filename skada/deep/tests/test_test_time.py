# Author : Maxence Barneche
#
# License: BSD-3-Clause

import pytest

torch = pytest.importorskip("torch")

from skada.datasets import make_shifted_datasets
from skada.deep._test_time import TestTimeCriterion, TestTimeNet
from skada.deep.base import DomainAwareModule, DomainBalancedDataLoader
from skada.deep.losses import TestLoss
from skada.deep.modules import ToyModule2D


def test_test_time_criterion():
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
    method = TestTimeNet(
        DomainAwareModule(module, "dropout"),
        iterator_train=DomainBalancedDataLoader,
        criterion=criterion,
        batch_size=10,
        max_epochs=2,
        train_split=None,
    )

    method.fit(X, y, sample_domain)
