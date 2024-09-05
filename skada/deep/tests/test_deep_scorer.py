# Author: Yanis Lalou <yanis.lalou@polytechnique.edu>
#
# License: BSD 3-Clause

import numpy as np
import pytest
import torch
from sklearn.model_selection import ShuffleSplit, cross_validate
from sklearn.preprocessing import StandardScaler

from skada import make_da_pipeline, source_target_split
from skada.deep import DeepCoral
from skada.deep.modules import ToyCNN, ToyModule2D
from skada.metrics import (
    CircularValidation,
    DeepEmbeddedValidation,
    MixValScorer,
    PredictionEntropyScorer,
    SoftNeighborhoodDensity,
)


@pytest.mark.parametrize(
    "scorer",
    [
        DeepEmbeddedValidation(),
        PredictionEntropyScorer(),
        SoftNeighborhoodDensity(),
        CircularValidation(),
        MixValScorer(),
    ],
)
def test_generic_scorer_on_deepmodel(scorer, da_dataset):
    X, y, sample_domain = da_dataset.pack_train(as_sources=["s"], as_targets=["t"])
    X_test, y_test, sample_domain_test = da_dataset.pack_test(as_targets=["t"])

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
        # DeepEmbeddedValidation(),
    ],
)
def test_generic_scorer(scorer, da_dataset):
    X, y, sample_domain = da_dataset.pack_train(as_sources=["s"], as_targets=["t"])

    net = DeepCoral(
        ToyModule2D(proba=True),
        reg=1,
        layer_name="dropout",
        batch_size=10,
        max_epochs=10,
        train_split=None,
    )
    estimator = make_da_pipeline(
        StandardScaler(),
        net,
    )
    cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
    scores = cross_validate(
        estimator,
        X.astype(np.float32),
        y,
        cv=cv,
        params={"sample_domain": sample_domain},
        scoring=scorer,
    )["test_score"]
    assert scores.shape[0] == 3, "evaluate 3 splits"
    assert np.all(~np.isnan(scores)), "all scores are computed"


def test_dev_cnn_features_nd(da_dataset):
    X, y, sample_domain = da_dataset.pack_train(as_sources=["s"], as_targets=["t"])
    X = np.repeat(X[..., np.newaxis], repeats=5, axis=-1)  # Make it batched 2D data
    X = X.astype(np.float32)

    scorer = DeepEmbeddedValidation()
    _, n_channels, input_size = X.shape
    y_source, _ = source_target_split(y, sample_domain=sample_domain)
    n_classes = len(np.unique(y_source))
    module = ToyCNN(
        n_channels=n_channels,
        input_size=input_size,
        n_classes=n_classes,
        kernel_size=3,
        out_channels=2,
    )
    # Assert features more than 2D
    assert module.feature_extractor(torch.tensor(X)).ndim > 2

    net = DeepCoral(
        module,
        reg=1,
        layer_name="feature_extractor",
        batch_size=10,
        max_epochs=10,
        train_split=None,
    )

    cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
    scores = cross_validate(
        net,
        X,
        y,
        cv=cv,
        params={"sample_domain": sample_domain},
        scoring=scorer,
    )["test_score"]
    assert scores.shape[0] == 3, "evaluate 3 splits"
    assert np.all(~np.isnan(scores)), "all scores are computed"
