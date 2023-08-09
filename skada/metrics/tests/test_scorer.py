# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD 3-Clause
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

from skada import ReweightDensity, SubspaceAlignment
from skada.metrics import (
    SupervisedScorer,
    ImportanceWeightedScorer,
    PredictionEntropyScorer,
    DeepEmbeddedValidation,
    SoftNeighborhoodDensity,
)
import pytest


@pytest.mark.parametrize(
    "scorer_name",
    [
        SupervisedScorer,
        ImportanceWeightedScorer,
        PredictionEntropyScorer,
        DeepEmbeddedValidation,
        SoftNeighborhoodDensity,
    ],
)
def test_scorer(scorer_name, tmp_da_dataset):
    X, y, X_target, y_target = tmp_da_dataset
    estimator = ReweightDensity(base_estimator=LogisticRegression())

    if scorer_name == SupervisedScorer:
        scorer = scorer_name(X_target, y_target)
    else:
        scorer = scorer_name(X_target)

    cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
    scores = cross_val_score(
        estimator,
        X,
        y,
        cv=cv,
        fit_params={"X_target": X_target},
        scoring=scorer,
    )
    assert len(scores) == 3
    assert scores[0] != np.nan


@pytest.mark.parametrize(
    "scorer_name",
    [
        PredictionEntropyScorer,
        SoftNeighborhoodDensity,
    ],
)
def test_scorer_with_entropy(scorer_name, tmp_da_dataset):
    X, y, X_target, y_target = tmp_da_dataset

    estimator = ReweightDensity(base_estimator=SVC())
    scorer = scorer_name(X_target)

    estimator.fit(X, y, X_target)
    with pytest.raises(AttributeError):
        scorer(estimator, X, y)


def test_scorer_with_log_proba():
    """ Test to ensure `BaseSubspaceEstimator` works correctly
    with base estimator that defines `predict_log_proba`.
    See https://github.com/tgnassou/da-toolbox/issues/48 for details.
    """
    n_samples, n_features = 100, 5
    rng = np.random.RandomState(42)
    X, y, X_target = (
        rng.rand(n_samples, n_features),
        rng.randint(2, size=n_samples),
        rng.rand(n_samples, n_features)
    )
    estimator = SubspaceAlignment(
        base_estimator=LogisticRegression(), n_components=2)
    scorer = PredictionEntropyScorer(X_target)

    cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
    scores = cross_val_score(
        estimator,
        X,
        y,
        cv=cv,
        fit_params={"X_target": X_target},
        scoring=scorer,
    )
    assert scores.shape[0] == 3, "evaluate 3 splits"
    assert np.all(~np.isnan(scores)), "all scores are computed"
