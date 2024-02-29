# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#
# License: BSD 3-Clause
import pytest

import numpy as np

torch = pytest.importorskip("torch")

from skada.deep.base import (
    DomainAwareModule,
    DomainAwareCriterion,
    DomainBalancedDataLoader,
    DomainAwareNet,
    BaseDALoss
)
from skada.deep.modules import ToyModule2D
from skada.datasets import make_shifted_datasets
from skorch.dataset import Dataset



def test_domainawaretraining():

    class TestLoss(BaseDALoss):
        """Test Loss to check the deep API"""

        def __init__(self,):
            super(TestLoss, self).__init__()

        def forward(
            self,
            y_s,
            y_pred_s,
            y_pred_t,
            domain_pred_s,
            domain_pred_t,
            features_s,
            features_t,
        ):
            """Compute the domain adaptation loss"""
            return 0

    module = ToyModule2D()
    module.eval()

    n_samples = 20
    dataset = make_shifted_datasets(
        n_samples_source=n_samples,
        n_samples_target=n_samples,
        shift="concept_drift",
        noise=0.1,
        random_state=42,
        return_dataset=True,
    )

    method = DomainAwareNet(
        DomainAwareModule(module, "dropout"),
        iterator_train=DomainBalancedDataLoader,
        criterion=DomainAwareCriterion(
            torch.nn.CrossEntropyLoss(), TestLoss()
        ),
        batch_size=10,
        max_epochs=2,
        train_split=None
    )

    X, y, sample_domain = dataset.pack_train(as_sources=["s"], as_targets=["t"])
    X_test, y_test, sample_domain_test = dataset.pack_test(as_targets=["t"])
    X = X.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # without dict
    method.fit(X, y, sample_domain)

    y_pred = method.predict(X_test, sample_domain_test)
    method.score(X_test, y_test, sample_domain_test)

    assert y_pred.shape[0] == X_test.shape[0]

    # with dict
    X_dict = {"X": X, "sample_domain": sample_domain}
    method.fit(X_dict, y,)

    X_dict_test = {"X": X_test, "sample_domain": sample_domain_test}

    y_pred = method.predict(X_dict_test)
    method.score(X_dict_test, y_test)

    assert y_pred.shape[0] == X_test.shape[0]

    # numpy input
    method.fit(X, y, sample_domain)
    y_pred = method.predict(X_test, sample_domain_test)
    method.score(X_test, y_test, sample_domain_test)

    assert y_pred.shape[0] == X_test.shape[0]

    # tensor input
    method.fit(torch.tensor(X), torch.tensor(y), torch.tensor(sample_domain))
    y_pred = method.predict(torch.tensor(X_test), torch.tensor(sample_domain_test))
    method.score(
        torch.tensor(X_test),
        torch.tensor(y_test),
        torch.tensor(sample_domain_test)
    )

    assert y_pred.shape[0] == X_test.shape[0]

    # dataset input
    X_dict = {"X": X, "sample_domain": sample_domain}

    torch_dataset = Dataset(X_dict, y)
    method.fit(torch_dataset, y=None)

    # Test keys name in the dict
    X, y, sample_domain = dataset.pack_train(as_sources=["s"], as_targets=["t"])
    X_dict = {"bad_name": X.astype(np.float32), "sample_domain": sample_domain}
    with pytest.raises(ValueError):
        method.fit(X_dict, y,)

    with pytest.raises(ValueError):
        method.predict(X_dict,)

    with pytest.raises(ValueError):
        method.score(X_dict, y,)

    X_dict = {"X": X.astype(np.float32), "bad_name": sample_domain}
    with pytest.raises(ValueError):
        method.fit(X_dict, y,)

    with pytest.raises(ValueError):
        method.predict(X_dict,)

    with pytest.raises(ValueError):
        method.score(X_dict, y,)

    # Test dataset without dict
    torch_dataset = Dataset(X, y)

    with pytest.raises(ValueError):
        method.fit(torch_dataset, y=None)

    with pytest.raises(ValueError):
        method.predict(torch_dataset,)


def test_domainbalanceddataloader():
    n_samples = 20
    dataset = make_shifted_datasets(
        n_samples_source=n_samples,
        n_samples_target=n_samples,
        shift="concept_drift",
        noise=0.1,
        random_state=42,
        return_dataset=True,
    )
    X, y, sample_domain = dataset.pack_train(as_sources=["s"], as_targets=["t"])
    X_dict = {"X": X.astype(np.float32), "sample_domain": sample_domain}

    dataset = Dataset(X_dict, y)

    dataloader = DomainBalancedDataLoader(dataset, batch_size=10)

    for batch in dataloader:
        X, y = batch
        sample_domain = X["sample_domain"]
        assert len(sample_domain > 0) == len(sample_domain < 0)
