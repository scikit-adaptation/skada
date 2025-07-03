# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#         Yanis Lalou <yanis.lalou@polytechnique.edu>
#
# License: BSD 3-Clause

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    LeaveOneGroupOut,
    cross_validate,
)

from skada import SubspaceAlignmentAdapter, make_da_pipeline
from skada.metrics import PredictionEntropyScorer
from skada.model_selection import (
    DomainShuffleSplit,
    LeaveOneDomainOut,
    SourceTargetShuffleSplit,
    StratifiedDomainShuffleSplit,
)


@pytest.mark.parametrize(
    "cv, n_splits",
    [
        (GroupShuffleSplit(n_splits=2, test_size=0.3, random_state=0), 2),
        (GroupKFold(n_splits=2), 2),
        (LeaveOneGroupOut(), 4),
    ],
)
def test_group_based_cv(da_dataset, cv, n_splits):
    assert hasattr(cv, "set_split_request"), "splitter has to support routing"
    cv.set_split_request(groups="sample_domain")
    # single source, single target, target labels are masked
    X, y, sample_domain = da_dataset.pack(
        as_sources=["s", "s2"], as_targets=["t", "t2"], mask_target_labels=True
    )
    # by default, each estimator in the pipeline is wrapped into `Shared` selector
    pipe = make_da_pipeline(
        SubspaceAlignmentAdapter(n_components=2),
        LogisticRegression(),
    )
    pipe.fit(X, y, sample_domain=sample_domain)
    scores = cross_validate(
        pipe,
        X,
        y,
        cv=cv,
        params={"sample_domain": sample_domain},
        scoring=PredictionEntropyScorer(),
    )["test_score"]
    assert scores.shape[0] == n_splits, "evaluate all splits"
    # xxx(okachaiev): make sure we understand why/when validation fails
    # (some results are certainly None in here)
    assert np.any(~np.isnan(scores)), "at least some scores are computed"


@pytest.mark.parametrize(
    "cv",
    [
        (SourceTargetShuffleSplit(n_splits=4, test_size=0.3, random_state=0)),
        (DomainShuffleSplit(n_splits=4, test_size=0.3, random_state=0)),
        (StratifiedDomainShuffleSplit(n_splits=4, test_size=0.3, random_state=0)),
    ],
)
def test_domain_aware_split(da_dataset, cv):
    X, y, sample_domain = da_dataset.pack(
        as_sources=["s", "s2"], as_targets=["t", "t2"], mask_target_labels=True
    )
    pipe = make_da_pipeline(
        SubspaceAlignmentAdapter(n_components=2),
        LogisticRegression(),
    )
    pipe.fit(X, y, sample_domain=sample_domain)
    scores = cross_validate(
        pipe,
        X,
        y,
        cv=cv,
        params={"sample_domain": sample_domain},
        scoring=PredictionEntropyScorer(),
    )["test_score"]
    assert scores.shape[0] == 4, "evaluate all splits"
    assert np.all(~np.isnan(scores)), "at least some scores are computed"


@pytest.mark.parametrize("max_n_splits, n_splits", [(2, 2), (10, 4)])
def test_leave_one_domain_out(da_dataset, max_n_splits, n_splits):
    X, y, sample_domain = da_dataset.pack_lodo()
    pipe = make_da_pipeline(
        SubspaceAlignmentAdapter(n_components=2),
        LogisticRegression(),
    )
    pipe.fit(X, y, sample_domain=sample_domain)
    cv = LeaveOneDomainOut(max_n_splits=max_n_splits, test_size=0.3, random_state=0)
    scores = cross_validate(
        pipe,
        X,
        y,
        cv=cv,
        params={"sample_domain": sample_domain},
        scoring=PredictionEntropyScorer(),
    )["test_score"]
    assert scores.shape[0] == n_splits, "evaluate all splits"
    assert np.all(~np.isnan(scores)), "all scores are computed"


def test_domain_shuffle_split(da_dataset):
    X, y, sample_domain = da_dataset.pack(
        as_sources=["s", "s2"], as_targets=["t", "t2"], mask_target_labels=True
    )
    pipe = make_da_pipeline(
        SubspaceAlignmentAdapter(n_components=2),
        LogisticRegression(),
    )

    with pytest.raises(ValueError):
        DomainShuffleSplit(n_splits=4, test_size=0.3, random_state=0, under_sampling=2)

    cv = DomainShuffleSplit(n_splits=4, test_size=0.3, random_state=0)
    scores = cross_validate(
        pipe,
        X,
        y,
        cv=cv,
        params={"sample_domain": sample_domain},
        scoring=PredictionEntropyScorer(),
    )["test_score"]
    assert scores.shape[0] == 4, "evaluate all splits"
    assert np.all(~np.isnan(scores)), "all scores are computed"


def test_stratified_domain_shuffle_split_exceptions():
    # Test with y.ndim == 2 nothing is raised
    X = np.ones((10, 2))
    y = np.array(5 * [[0], [1]])
    sample_domain = np.array(5 * [0, 1])
    splitter = StratifiedDomainShuffleSplit(n_splits=4, test_size=0.5, random_state=0)
    next(iter(splitter.split(X, y, sample_domain)))

    # Test np.min(group_counts) < 2
    X = np.ones((2, 2))
    y = np.array([0, 1])
    sample_domain = np.array([0, 1])
    splitter = StratifiedDomainShuffleSplit(n_splits=4, test_size=0.5, random_state=0)
    with pytest.raises(ValueError):
        next(iter(splitter.split(X, y, sample_domain)))

    # Test n_train < n_groups:
    X = np.ones((10, 2))
    y = np.array(5 * [0, 1])
    sample_domain = np.array(5 * [0, 1])
    splitter = StratifiedDomainShuffleSplit(n_splits=4, test_size=0.9, random_state=0)
    with pytest.raises(ValueError):
        next(iter(splitter.split(X, y, sample_domain)))

    # Test n_test < n_groups:
    X = np.ones((10, 2))
    y = np.array(5 * [0, 1])
    sample_domain = np.array(5 * [0, 1])
    splitter = StratifiedDomainShuffleSplit(n_splits=4, test_size=0.1, random_state=0)
    with pytest.raises(ValueError):
        next(iter(splitter.split(X, y, sample_domain)))


@pytest.mark.parametrize(
    "cv",
    [
        (GroupShuffleSplit(n_splits=2, test_size=0.3, random_state=0)),
        (GroupKFold(n_splits=2)),
        (LeaveOneGroupOut()),
        (SourceTargetShuffleSplit(n_splits=2, test_size=0.3, random_state=0)),
        (
            DomainShuffleSplit(
                n_splits=2, test_size=0.3, random_state=0, under_sampling=1
            )
        ),
        (StratifiedDomainShuffleSplit(n_splits=2, test_size=0.3, random_state=0)),
    ],
)
def test_cv_with_nd_dimensional_X(da_dataset, cv):
    X, y, sample_domain = da_dataset.pack_lodo()
    # Transform X from 2D to 3D
    X = X.reshape(X.shape[0], -1, 1)  # Reshape to (n_samples, n_features, 1)
    assert X.ndim == 3, "X should be 3-dimensional after reshaping"

    splits = list(cv.split(X, y, sample_domain))

    for train, test in splits:
        assert isinstance(train, np.ndarray) and isinstance(
            test, np.ndarray
        ), "split indices should be numpy arrays"
        assert len(train) + len(test) == len(
            X
        ), "train and test indices should cover all samples"
        assert (
            len(np.intersect1d(train, test)) == 0
        ), "train and test indices should not overlap"
