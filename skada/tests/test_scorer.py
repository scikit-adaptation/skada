# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#         Yanis Lalou <yanis.lalou@polytechnique.edu>
#         Ambroise Odonnat <ambroiseodonnattechnologie@gmail.com>
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
    MaNoScorer,
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
        MaNoScorer(),
    ],
)
def test_generic_scorer(scorer, da_dataset):
    X, y, sample_domain = da_dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )
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
    X, y, sample_domain = da_dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )
    estimator = make_da_pipeline(
        DensityReweightAdapter(),
        LogisticRegression()
        .set_fit_request(sample_weight=True)
        .set_score_request(sample_weight=True),
    )
    cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
    _, target_labels, _ = da_dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=False
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
        MaNoScorer(),
    ],
)
def test_scorer_with_entropy_requires_predict_proba(scorer, da_dataset):
    X, y, sample_domain = da_dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )
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
    X, y, sample_domain = dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )
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
    X, y, sample_domain = da_dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )
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
    X, y, sample_domain = da_dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )
    _, unmasked_y, _ = da_dataset.pack(
        as_sources=[], as_targets=["t"], mask_target_labels=False
    )
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

    X, y, sample_domain = dataset_reg.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )
    _, unmasked_y, _ = dataset_reg.pack(
        as_sources=[], as_targets=["t"], mask_target_labels=False
    )

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
    X, y, sample_domain = da_dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )
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
    X, y, sample_domain = da_dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )
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

    # Test for ice_diff or ice_same = NaN
    class DummyEstimator:
        def __init__(self, ice_type):
            self.ice_type = ice_type

        def predict(self, X, sample_domain=None):
            if self.ice_type == "intra":
                return np.arange(X.shape[0])  # All predictions are different
            elif self.ice_type == "inter":
                return np.zeros(X.shape[0])  # All predictions are the same

        def fit(self, X, y, sample_domain=None):
            return self

        def score(self, X, y, sample_domain=None):
            return 1.0  # Always return a perfect score

    dummy_estimator = DummyEstimator(ice_type="intra")

    # Test intra-cluster case (ice_same should be NaN)
    scorer_intra = MixValScorer(alpha=0.55, random_state=42, ice_type="intra")
    score_intra = scorer_intra._score(dummy_estimator, X, y, sample_domain)
    assert np.isnan(
        score_intra
    ), "intra-cluster score should be NaN when all predictions are the same"

    dummy_estimator = DummyEstimator(ice_type="inter")

    # Test inter-cluster case (ice_diff should be NaN)
    scorer_inter = MixValScorer(alpha=0.55, random_state=42, ice_type="inter")
    score_inter = scorer_inter._score(dummy_estimator, X, y, sample_domain)
    assert np.isnan(
        score_inter
    ), "inter-cluster score should be NaN when all predictions are the same"

    # Test both case with score_inter == Nan (result should be a number)
    scorer_both = MixValScorer(alpha=0.55, random_state=42, ice_type="both")
    score_both = scorer_both._score(dummy_estimator, X, y, sample_domain)
    assert not np.isnan(
        score_both
    ), "combined score should not be NaN when both intra and inter scores are NaN"


def test_mixval_scorer_regression(da_reg_dataset):
    X, y, sample_domain = da_reg_dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=False
    )

    estimator = make_da_pipeline(DensityReweightAdapter(), LinearRegression())

    scorer = MixValScorer(alpha=0.55, random_state=42)
    with pytest.raises(ValueError):
        scorer(estimator, X, y, sample_domain)


def test_mano_scorer(da_dataset):
    X, y, sample_domain = da_dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )
    estimator = make_da_pipeline(
        DensityReweightAdapter(),
        LogisticRegression().set_fit_request(sample_weight=True),
    )

    estimator.fit(X, y, sample_domain=sample_domain)

    scorer = MaNoScorer()
    score_mean = scorer._score(estimator, X, y, sample_domain=sample_domain)
    assert isinstance(score_mean, float), "score_mean is not a float"

    # Test softmax normalization
    scorer = MaNoScorer(threshold=-1)
    score_mean = scorer._score(estimator, X, y, sample_domain=sample_domain)
    assert isinstance(score_mean, float), "score_mean is not a float"

    # Test softmax normalization
    scorer = MaNoScorer(threshold=float("inf"))
    score_mean = scorer._score(estimator, X, y, sample_domain=sample_domain)
    assert isinstance(score_mean, float), "score_mean is not a float"

    # Test invalid p-norm order
    with pytest.raises(ValueError):
        MaNoScorer(p=-1)

    # Test correct output range
    scorer = MaNoScorer()
    score_mean = scorer._score(estimator, X, y, sample_domain=sample_domain)
    assert (scorer._sign * score_mean >= 0) and (
        scorer._sign * score_mean <= 1
    ), "The output range should be [-1, 0] or [0, 1]."


def test_mano_scorer_regression(da_reg_dataset):
    X, y, sample_domain = da_reg_dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=False
    )

    estimator = make_da_pipeline(DensityReweightAdapter(), LogisticRegression())

    scorer = MaNoScorer()
    with pytest.raises(ValueError):
        scorer(estimator, X, y, sample_domain)


@pytest.mark.parametrize(
    "scorer",
    [
        SupervisedScorer(),
        ImportanceWeightedScorer(),
        PredictionEntropyScorer(),
        SoftNeighborhoodDensity(),
        CircularValidation(),
        MixValScorer(alpha=0.55, random_state=42),
        MaNoScorer(),
    ],
)
def test_scorer_with_nd_input(scorer, da_dataset):
    X, y, sample_domain = da_dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )

    # Repeat data to have a 3D input
    X_3d = np.repeat(X[:, :, None], repeats=3, axis=2)

    estimator = make_da_pipeline(
        DummyClassifier(strategy="stratified", random_state=42)
        .set_fit_request(sample_weight=True)
        .set_score_request(sample_weight=True),
    )
    cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
    if isinstance(scorer, SupervisedScorer):
        _, target_labels, _ = da_dataset.pack(
            as_sources=["s"], as_targets=["t"], mask_target_labels=False
        )
        params = {"sample_domain": sample_domain, "target_labels": target_labels}
    else:
        params = {"sample_domain": sample_domain}
    scores = cross_validate(
        estimator,
        X_3d,
        y,
        cv=cv,
        params=params,
        scoring=scorer,
    )["test_score"]

    assert scores.shape[0] == 3, "evaluate 3 splits"
    assert np.all(~np.isnan(scores)), "all scores are computed"
