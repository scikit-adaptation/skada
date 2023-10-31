import numpy as np
import sklearn
sklearn.set_config(enable_metadata_routing=True)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit, cross_validate

from skada import (
    DomainAwareEstimator,
    ReweightDensityAdapter,
    SubspaceAlignmentAdapter,
)
from skada.datasets import DomainAwareDataset
from skada.metrics import (
    SupervisedScorer,
    ImportanceWeightedScorer,
    PredictionEntropyScorer,
    DeepEmbeddedValidation,
    SoftNeighborhoodDensity,
)

import pytest


@pytest.mark.parametrize(
    "scorer",
    [
        SupervisedScorer(),
        # ImportanceWeightedScorer(),
        # PredictionEntropyScorer(),
        # DeepEmbeddedValidation(),
        # SoftNeighborhoodDensity(),
    ],
)
def test_scorer(scorer, da_dataset):
    X, y, sample_domain = da_dataset.pack(as_sources=['s'], as_targets=['t'])
    estimator = DomainAwareEstimator(ReweightDensityAdapter(), LogisticRegression())
    cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
    scores = cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        params={'sample_domain': sample_domain},
        scoring=scorer,
    )['test_score']
    assert scores.shape[0] == 3, "evaluate 3 splits"
    assert np.all(~np.isnan(scores)), "all scores are computed"


@pytest.mark.parametrize(
    "scorer",
    [
        PredictionEntropyScorer(),
        SoftNeighborhoodDensity(),
    ],
)
def test_scorer_with_entropy_requires_predict_proba(scorer, da_dataset):
    X, y, sample_domain = da_dataset.pack(as_sources=['s'], as_targets=['t'])
    estimator = DomainAwareEstimator(ReweightDensityAdapter(), SVC())
    estimator.fit(X, y, sample_domain)
    with pytest.raises(AttributeError):
        scorer(estimator, X, y, sample_domain=sample_domain)


def test_scorer_with_log_proba():
    n_samples, n_features = 100, 5
    rng = np.random.RandomState(42)
    dataset = DomainAwareDataset(domains=[
        (rng.rand(n_samples, n_features), rng.randint(2, size=n_samples), 's'),
        (rng.rand(n_samples, n_features), None, 't')
    ])
    X, y, sample_domain = dataset.pack(as_sources=['s'], as_targets=['t'])
    estimator = DomainAwareEstimator(
        SubspaceAlignmentAdapter(n_components=2),
        LogisticRegression()
    )
    cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
    scores = cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        params={'sample_domain': sample_domain},
        scoring=PredictionEntropyScorer(),
    )['test_score']
    assert scores.shape[0] == 3, "evaluate 3 splits"
    assert np.all(~np.isnan(scores)), "all scores are computed"
