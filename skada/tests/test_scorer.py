# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#         Yanis Lalou <yanis.lalou@polytechnique.edu>
#
# License: BSD 3-Clause

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ShuffleSplit, cross_validate
from sklearn.svm import SVC

from skada import (
    DensityReweightAdapter,
    SubspaceAlignmentAdapter,
    make_da_pipeline,
)
from skada.datasets import DomainAwareDataset, make_shifted_datasets
from skada.metrics import (
    CircularValidation,
    DeepEmbeddedValidation,
    ImportanceWeightedScorer,
    MixValScorer,
    PredictionEntropyScorer,
    SoftNeighborhoodDensity,
    SupervisedScorer,
)


@pytest.mark.parametrize(
    "scorer",
    [
        ImportanceWeightedScorer(),
        PredictionEntropyScorer(),
        SoftNeighborhoodDensity(),
        DeepEmbeddedValidation(),
        CircularValidation(),
    ],
)
def test_generic_scorer(scorer, da_dataset):
    X, y, sample_domain = da_dataset.pack_train(as_sources=["s"], as_targets=["t"])
    estimator = make_da_pipeline(
        DensityReweightAdapter(),
        LogisticRegression()
        .set_fit_request(sample_weight=True)
        .set_score_request(sample_weight=True),
    )
    cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
    scores = cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        params={"sample_domain": sample_domain},
        scoring=scorer,
    )["test_score"]
    assert scores.shape[0] == 3, "evaluate 3 splits"
    assert np.all(~np.isnan(scores)), "all scores are computed"


def test_supervised_scorer(da_dataset):
    """`SupervisedScorer` requires unmasked target label to be available."""
    X, y, sample_domain = da_dataset.pack_train(as_sources=["s"], as_targets=["t"])
    estimator = make_da_pipeline(
        DensityReweightAdapter(),
        LogisticRegression()
        .set_fit_request(sample_weight=True)
        .set_score_request(sample_weight=True),
    )
    cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
    _, target_labels, _ = da_dataset.pack(
        as_sources=["s"], as_targets=["t"], train=False
    )
    scoring = SupervisedScorer()
    scores = cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        params={"sample_domain": sample_domain, "target_labels": target_labels},
        scoring=scoring,
    )["test_score"]
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
    X, y, sample_domain = da_dataset.pack_train(as_sources=["s"], as_targets=["t"])
    estimator = make_da_pipeline(
        DensityReweightAdapter(), SVC().set_fit_request(sample_weight=True)
    )
    estimator.fit(X, y, sample_domain=sample_domain)
    with pytest.raises(AttributeError):
        scorer(estimator, X, y, sample_domain=sample_domain)


def test_scorer_with_log_proba():
    n_samples, n_features = 100, 5
    rng = np.random.default_rng(42)
    dataset = DomainAwareDataset(
        domains=[
            (
                rng.standard_normal((n_samples, n_features)),
                rng.integers(2, size=n_samples),
                "s",
            ),
            (rng.standard_normal((n_samples, n_features)), None, "t"),
        ]
    )
    X, y, sample_domain = dataset.pack_train(as_sources=["s"], as_targets=["t"])
    estimator = make_da_pipeline(
        SubspaceAlignmentAdapter(n_components=2), LogisticRegression()
    )
    cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
    scores = cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        params={"sample_domain": sample_domain},
        scoring=PredictionEntropyScorer(),
    )["test_score"]
    assert scores.shape[0] == 3, "evaluate 3 splits"
    assert np.all(~np.isnan(scores)), "all scores are computed"
    assert np.all(scores <= 0), "all scores are negative"


def test_prediction_entropy_scorer_reduction(da_dataset):
    X, y, sample_domain = da_dataset.pack_train(as_sources=["s"], as_targets=["t"])
    estimator = make_da_pipeline(
        DensityReweightAdapter(),
        LogisticRegression().set_fit_request(sample_weight=True),
    )

    estimator.fit(X, y, sample_domain=sample_domain)

    scorer = PredictionEntropyScorer(reduction="mean")
    score_mean = scorer._score(estimator, X, y, sample_domain=sample_domain)
    assert isinstance(score_mean, float), "score_mean is not a float"

    scorer = PredictionEntropyScorer(reduction="sum")
    score_sum = scorer._score(estimator, X, y, sample_domain=sample_domain)
    assert isinstance(score_sum, float), "score_sum is not a float"

    assert score_mean == pytest.approx(score_sum / X.shape[0], rel=1e-5)

    scorer = PredictionEntropyScorer(reduction="none")
    score_none = scorer._score(estimator, X, y, sample_domain=sample_domain)
    assert isinstance(score_none, np.ndarray), "score_none is not a numpy array"

    with pytest.raises(ValueError):
        scorer = PredictionEntropyScorer(reduction="WRONG_REDUCTION")

    # Really unlikely to happen, but still
    with pytest.raises(ValueError):
        scorer = PredictionEntropyScorer(reduction="none")
        scorer.reduction = "WRONG_REDUCTION"
        scorer._score(estimator, X, y, sample_domain=sample_domain)


def test_circular_validation(da_dataset):
    X, y, sample_domain = da_dataset.pack_train(as_sources=["s"], as_targets=["t"])
    _, unmasked_y, _ = da_dataset.pack_test(as_targets=["t"])
    estimator = make_da_pipeline(
        DensityReweightAdapter(),
        LogisticRegression().set_fit_request(sample_weight=True),
    )

    estimator.fit(X, y, sample_domain=sample_domain)

    scorer = CircularValidation()

    with pytest.raises(ValueError):
        scorer._score(estimator, X, unmasked_y, sample_domain=sample_domain)

    score = scorer._score(estimator, X, y, sample_domain=sample_domain)
    assert ~np.isnan(score), "the score is computed"

    # Test not callable source_scorer
    with pytest.raises(ValueError):
        scorer = CircularValidation(source_scorer=None)

    # Test unique y_pred_target
    estimator_dummy = make_da_pipeline(
        DummyRegressor(strategy="constant", constant=1),
    )
    estimator_dummy.fit(X, y, sample_domain=sample_domain)
    scorer = CircularValidation()
    score = scorer._score(estimator_dummy, X, y, sample_domain=sample_domain)
    assert ~np.isnan(score), "the score is computed"

    # Test regression task
    dataset_reg = make_shifted_datasets(
        n_samples_source=10,
        n_samples_target=10,
        noise=None,
        label="regression",
        return_dataset=True,
    )

    X, y, sample_domain = dataset_reg.pack_train(as_sources=["s"], as_targets=["t"])
    _, unmasked_y, _ = dataset_reg.pack_test(as_targets=["t"])

    estimator_regression = make_da_pipeline(
        DensityReweightAdapter(),
        LinearRegression().set_fit_request(sample_weight=True),
    )
    estimator_regression.fit(X, y, sample_domain=sample_domain)

    scorer = CircularValidation(
        source_scorer=mean_squared_error, greater_is_better=False
    )

    with pytest.raises(ValueError):
        scorer._score(estimator_regression, X, unmasked_y, sample_domain=sample_domain)

    score = scorer._score(estimator_regression, X, y, sample_domain=sample_domain)
    assert ~np.isnan(score), "the score is computed"


def test_deep_embedding_validation_no_transform(da_dataset):
    # Test that the scorer runs
    # even if the adapter does not have a `transform` method

    scorer = DeepEmbeddedValidation()
    X, y, sample_domain = da_dataset.pack_train(as_sources=["s"], as_targets=["t"])
    estimator = make_da_pipeline(LogisticRegression())

    cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
    scores = cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        params={"sample_domain": sample_domain},
        scoring=scorer,
    )["test_score"]
    assert scores.shape[0] == 3, "evaluate 3 splits"
    assert np.all(~np.isnan(scores)), "all scores are computed"


def test_mixval_scorer(da_dataset):
    X, y, sample_domain = da_dataset.pack_train(as_sources=["s"], as_targets=["t"])
    estimator = make_da_pipeline(
        DensityReweightAdapter(),
        LogisticRegression()
        .set_fit_request(sample_weight=True)
        .set_score_request(sample_weight=True),
    )
    cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)

    # Test with default parameters
    scorer = MixValScorer(alpha=0.55, random_state=42)
    scores = cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        params={"sample_domain": sample_domain},
        scoring=scorer,
    )["test_score"]

    assert scores.shape[0] == 3, "evaluate 3 splits"
    assert np.all(~np.isnan(scores)), "all scores are computed"
    assert np.all(scores >= 0) and np.all(scores <= 1), "scores are between 0 and 1"

    # Test different ice_type options
    for ice_type in ["both", "intra", "inter"]:
        scorer = MixValScorer(alpha=0.55, random_state=42, ice_type=ice_type)
        scores = cross_validate(
            estimator,
            X,
            y,
            cv=cv,
            params={"sample_domain": sample_domain},
            scoring=scorer,
        )["test_score"]

        assert scores.shape[0] == 3, f"evaluate 3 splits for ice_type={ice_type}"
        assert np.all(
            ~np.isnan(scores)
        ), f"all scores are computed for ice_type={ice_type}"
        assert np.all(scores >= 0) and np.all(
            scores <= 1
        ), f"scores are between 0 and 1 for ice_type={ice_type}"

    # Test invalid ice_type
    with pytest.raises(ValueError):
        MixValScorer(ice_type="invalid")


def test_mixval_scorer_regression(da_reg_dataset):
    X, y, sample_domain = da_reg_dataset

    estimator = make_da_pipeline(DensityReweightAdapter(), LinearRegression())

    scorer = MixValScorer(alpha=0.55, random_state=42)
    with pytest.raises(ValueError):
        scorer(estimator, X, y, sample_domain)


@pytest.mark.parametrize(
    "scorer",
    [
        ImportanceWeightedScorer(),
        # PredictionEntropyScorer(),
        SoftNeighborhoodDensity(),
        # DeepEmbeddedValidation(),
        # CircularValidation(),
        # MixValScorer(alpha=0.55, random_state=42),
    ],
)
def test_scorer_with_nd_input(scorer, da_dataset):
    X, y, sample_domain = da_dataset.pack_train(as_sources=["s"], as_targets=["t"])

    # Reshape X to be 3D
    X_3d = X.reshape(X.shape[0], -1, 1)

    estimator = make_da_pipeline(
        DummyClassifier(strategy="stratified", random_state=42)
        .set_fit_request(sample_weight=True)
        .set_score_request(sample_weight=True),
    )
    cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
    scores = cross_validate(
        estimator,
        X_3d,
        y,
        cv=cv,
        params={"sample_domain": sample_domain},
        scoring=scorer,
        error_score="raise",
    )["test_score"]

    assert scores.shape[0] == 3, "evaluate 3 splits"
    assert np.all(~np.isnan(scores)), "all scores are computed"
