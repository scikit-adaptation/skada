# Author: Yanis Lalou <yanis.lalou@polytechnique.edu>
#
# License: BSD 3-Clause

import numpy as np
import pytest
import torch

from skada.deep.base import (
    BaseDALoss,
    DomainAwareCriterion,
    DomainAwareModule,
    DomainAwareNet,
    DomainBalancedDataLoader,
)
from skada.deep.modules import ToyModule2D
from skada.metrics import (
    DeepEmbeddedValidation,
)


class TestLoss(BaseDALoss):
    """Test Loss to check the deep API"""

    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self,
        *args,
    ):
        """Compute the domain adaptation loss"""
        return 0


@pytest.mark.parametrize(
    "scorer",
    [
        DeepEmbeddedValidation(),
    ],
)
def test_generic_scorer_on_deepmodel(scorer, da_dataset):
    X, y, sample_domain = da_dataset.pack_train(as_sources=["s"], as_targets=["t"])
    X_test, y_test, sample_domain_test = da_dataset.pack_test(as_targets=["t"])

    module = ToyModule2D()

    estimator = DomainAwareNet(
        DomainAwareModule(module, "dropout"),
        iterator_train=DomainBalancedDataLoader,
        criterion=DomainAwareCriterion(torch.nn.CrossEntropyLoss(), TestLoss()),
        batch_size=10,
        max_epochs=2,
        train_split=None,
    )

    X = X.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # without dict
    estimator.fit(X, y, sample_domain=sample_domain)

    estimator.predict(X_test, sample_domain=sample_domain_test)
    estimator.predict_proba(X, sample_domain=sample_domain)

    scores = scorer._score(estimator, X, y, sample_domain)

    assert ~np.isnan(scores), "The score is computed"
