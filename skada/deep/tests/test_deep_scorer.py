# Author: Yanis Lalou <yanis.lalou@polytechnique.edu>
#         Ambroise Odonnat <ambroiseodonnattechnologie@gmail.com>
#
# License: BSD 3-Clause

import numpy as np
import pytest
import torch
from sklearn.model_selection import ShuffleSplit, cross_validate
from sklearn.preprocessing import StandardScaler

from skada import make_da_pipeline, source_target_split
from skada.deep import DeepCoral
from skada.deep._baseline import SourceOnly, TargetOnly
from skada.deep.modules import ToyCNN, ToyModule2D
from skada.metrics import (
    CircularValidation,
    DeepEmbeddedValidation,
    ImportanceWeightedScorer,
    MaNoScorer,
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
        MaNoScorer(),
        MixValScorer(),
        ImportanceWeightedScorer(),
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

    estimator.predict(X_test, sample_domain=sample_domain_test, allow_source=True)
    estimator.predict_proba(X, sample_domain=sample_domain, allow_source=True)

    scores = scorer(estimator, X, y, sample_domain)

    assert ~np.isnan(scores), "The score is computed"


@pytest.mark.parametrize(
    "scorer",
    [
        PredictionEntropyScorer(),
        SoftNeighborhoodDensity(),
        DeepEmbeddedValidation(),
        MaNoScorer(),
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


@pytest.mark.parametrize(
    "scorer",
    [
        DeepEmbeddedValidation(),
        ImportanceWeightedScorer(),
        MaNoScorer(),
    ],
)
def test_scorer_with_nd_features(scorer, da_dataset):
    X, y, sample_domain = da_dataset.pack_train(as_sources=["s"], as_targets=["t"])
    X = np.repeat(X[..., np.newaxis], repeats=5, axis=-1)  # Make it batched 2D data
    X = X.astype(np.float32)

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


def test_dev_scorer_on_target_only(da_dataset):
    X, y, sample_domain = da_dataset.pack_train(as_sources=["s"], as_targets=["t"])
    X_test, y_test, sample_domain_test = da_dataset.pack_test(as_targets=["t"])
    unmasked_y = np.copy(y)
    unmasked_y[sample_domain < 0] = y_test

    estimator = TargetOnly(
        ToyModule2D(proba=True),
        layer_name="dropout",
        batch_size=10,
        max_epochs=10,
        train_split=None,
    )

    X = X.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # without dict
    estimator.fit(X, unmasked_y, sample_domain=sample_domain)

    scores = DeepEmbeddedValidation()(estimator, X, unmasked_y, sample_domain)

    assert ~np.isnan(scores), "The score is computed"


def test_dev_scorer_on_source_only(da_dataset):
    X, y, sample_domain = da_dataset.pack_train(as_sources=["s"], as_targets=["t"])
    X_test, y_test, sample_domain_test = da_dataset.pack_test(as_targets=["t"])

    estimator = SourceOnly(
        ToyModule2D(proba=True),
        layer_name="dropout",
        batch_size=10,
        max_epochs=10,
        train_split=None,
    )

    X = X.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # without dict
    estimator.fit(X, y, sample_domain=sample_domain)

    scores = DeepEmbeddedValidation()(estimator, X, y, sample_domain)

    assert ~np.isnan(scores), "The score is computed"


def test_dev_exception_layer_name(da_dataset):
    X, y, sample_domain = da_dataset.pack_train(as_sources=["s"], as_targets=["t"])
    X_test, y_test, sample_domain_test = da_dataset.pack_test(as_targets=["t"])

    estimator = SourceOnly(
        ToyModule2D(proba=True),
        batch_size=10,
        max_epochs=10,
        train_split=None,
    )

    X = X.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # without dict
    estimator.fit(X, y, sample_domain=sample_domain)

    with pytest.raises(ValueError, match="The layer_name of the estimator is not set."):
        DeepEmbeddedValidation()(estimator, X, y, sample_domain)
