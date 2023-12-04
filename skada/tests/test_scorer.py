# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#
# License: BSD 3-Clause

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit, cross_validate
from sklearn.svm import SVC

from skada import (
    ReweightDensityAdapter,
    SubspaceAlignmentAdapter,
    make_da_pipeline,
)
from skada.datasets import DomainAwareDataset
from skada.metrics import (
    SupervisedScorer,
    ImportanceWeightedScorer,
    PredictionEntropyScorer,
    SoftNeighborhoodDensity,
)

import pytest


@pytest.mark.parametrize(
    "scorer",
    [
        ImportanceWeightedScorer(),
        PredictionEntropyScorer(),
        SoftNeighborhoodDensity(),
    ],
)
def test_generic_scorer(scorer, da_dataset):
    X, y, sample_domain = da_dataset.pack_train(as_sources=['s'], as_targets=['t'])
    estimator = make_da_pipeline(
        ReweightDensityAdapter(),
        LogisticRegression().set_score_request(sample_weight=True),
    )
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


def test_supervised_scorer(da_dataset):
    """`SupervisedScorer` requires unmasked target label to be available."""
    X, y, sample_domain = da_dataset.pack_train(as_sources=['s'], as_targets=['t'])
    estimator = make_da_pipeline(
        ReweightDensityAdapter(),
        LogisticRegression().set_score_request(sample_weight=True),
    )
    cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
    _, target_labels, _ = da_dataset.pack(
        as_sources=['s'],
        as_targets=['t'],
        train=False
    )
    scoring = SupervisedScorer()
    scores = cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        params={'sample_domain': sample_domain, 'target_labels': target_labels},
        scoring=scoring,
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
    X, y, sample_domain = da_dataset.pack_train(as_sources=['s'], as_targets=['t'])
    estimator = make_da_pipeline(ReweightDensityAdapter(), SVC())
    estimator.fit(X, y, sample_domain=sample_domain)
    with pytest.raises(AttributeError):
        scorer(estimator, X, y, sample_domain=sample_domain)


def test_scorer_with_log_proba():
    n_samples, n_features = 100, 5
    rng = np.random.RandomState(42)
    dataset = DomainAwareDataset(domains=[
        (rng.rand(n_samples, n_features), rng.randint(2, size=n_samples), 's'),
        (rng.rand(n_samples, n_features), None, 't')
    ])
    X, y, sample_domain = dataset.pack_train(as_sources=['s'], as_targets=['t'])
    estimator = make_da_pipeline(
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
