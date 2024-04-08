# Author: Yanis Lalou <yanis.lalou@polytechnique.edu>
#
# License: BSD 3-Clause

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit, cross_validate

from skada import make_da_pipeline
from skada.deep.base import (
    DomainAwareCriterion,
    DomainAwareModule,
    DomainAwareNet,
    DomainBalancedDataLoader,
)
from skada.deep.losses import TestLoss
from skada.deep.modules import ToyModule2D
from skada.metrics import (
    DeepEmbeddedValidation,
    PredictionEntropyScorer,
    SoftNeighborhoodDensity,
)
from skada.deep import DeepCoral


@pytest.mark.parametrize(
    "scorer",
    [
        DeepEmbeddedValidation(),
        PredictionEntropyScorer(),
        SoftNeighborhoodDensity(),
    ],
)
def test_generic_scorer_on_deepmodel(scorer, da_dataset):
    X, y, sample_domain = da_dataset.pack_train(as_sources=["s"], as_targets=["t"])
    X_test, y_test, sample_domain_test = da_dataset.pack_test(as_targets=["t"])

    module = ToyModule2D(proba=True)

    estimator = DomainAwareNet(
        DomainAwareModule(module, "dropout"),
        iterator_train=DomainBalancedDataLoader,
        criterion=DomainAwareCriterion(torch.nn.CrossEntropyLoss(), TestLoss()),
        batch_size=10,
        max_epochs=2,
        train_split=None,
    )

    estimator = DeepCoral(
        ToyModule2D(proba=True),
        reg=1,
        layer_name="dropout",
        batch_size=10,
        max_epochs=10,
        train_split=None,
    )

    X = X.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # without dict
    estimator.fit(X, y, sample_domain=sample_domain)

    estimator.predict(X_test, sample_domain=sample_domain_test)
    estimator.predict_proba(X, sample_domain=sample_domain)

    scores = scorer(estimator, X, y, sample_domain)

    assert ~np.isnan(scores), "The score is computed"


@pytest.mark.parametrize(
    "scorer",
    [
        PredictionEntropyScorer(),
        SoftNeighborhoodDensity(),
        DeepEmbeddedValidation(),
    ],
)
def test_generic_scorer(scorer, da_dataset):
    X, y, sample_domain = da_dataset.pack_train(as_sources=["s"], as_targets=["t"])
    estimator = make_da_pipeline(
        StandardScaler(),
        DeepCoral(
            ToyModule2D(),
            reg=1,
            layer_name="dropout",
            batch_size=10,
            max_epochs=10,
            train_split=None,
        ),
    )
    cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
    scores = cross_validate(
        estimator,
        X.astype(np.float32),
        y,
        cv=cv,
        params={"sample_domain": sample_domain},
        scoring=scorer,
    )['test_score']
    assert scores.shape[0] == 3, "evaluate 3 splits"
    assert np.all(~np.isnan(scores)), "all scores are computed"
