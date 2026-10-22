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
from sklearn.model_selection import ShuffleSplit, cross_val_score, cross_validate
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
def test_generic_scorer_with_cross_validate(scorer, da_dataset):
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
def test_generic_scorer_with_cross_val_score(scorer, da_dataset):
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
    scores = cross_val_score(
        estimator,
        X,
        y,
        cv=cv,
        params={"sample_domain": sample_domain},
        scoring=scorer,
    )
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
        error_score="raise",
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
    score_mean = scorer(estimator, X, y, sample_domain=sample_domain)
    assert isinstance(score_mean, float), "score_mean is not a float"

    scorer = PredictionEntropyScorer(reduction="sum")
    score_sum = scorer(estimator, X, y, sample_domain=sample_domain)
    assert isinstance(score_sum, float), "score_sum is not a float"

    assert score_mean == pytest.approx(score_sum / X.shape[0], rel=1e-5)

    scorer = PredictionEntropyScorer(reduction="none")
    score_none = scorer(estimator, X, y, sample_domain=sample_domain)
    assert isinstance(score_none, np.ndarray), "score_none is not a numpy array"

    with pytest.raises(ValueError):
        scorer = PredictionEntropyScorer(reduction="WRONG_REDUCTION")

    # Really unlikely to happen, but still
    with pytest.raises(ValueError):
        scorer = PredictionEntropyScorer(reduction="none")
        scorer.reduction = "WRONG_REDUCTION"
        scorer(estimator, X, y, sample_domain=sample_domain)


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
        scorer(estimator, X, unmasked_y, sample_domain=sample_domain)

    score = scorer(estimator, X, y, sample_domain=sample_domain)
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
    score = scorer(estimator_dummy, X, y, sample_domain=sample_domain)
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
        scorer(estimator_regression, X, unmasked_y, sample_domain=sample_domain)

    score = scorer(estimator_regression, X, y, sample_domain=sample_domain)
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
    score_intra = scorer_intra(dummy_estimator, X, y, sample_domain)
    assert np.isnan(
        score_intra
    ), "intra-cluster score should be NaN when all predictions are the same"

    dummy_estimator = DummyEstimator(ice_type="inter")

    # Test inter-cluster case (ice_diff should be NaN)
    scorer_inter = MixValScorer(alpha=0.55, random_state=42, ice_type="inter")
    score_inter = scorer_inter(dummy_estimator, X, y, sample_domain)
    assert np.isnan(
        score_inter
    ), "inter-cluster score should be NaN when all predictions are the same"

    # Test both case with score_inter == Nan (result should be a number)
    scorer_both = MixValScorer(alpha=0.55, random_state=42, ice_type="both")
    score_both = scorer_both(dummy_estimator, X, y, sample_domain)
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
    score_mean = scorer(estimator, X, y, sample_domain=sample_domain)
    assert isinstance(score_mean, float), "score_mean is not a float"

    # Test softmax normalization
    scorer = MaNoScorer(threshold=-1)
    score_mean = scorer(estimator, X, y, sample_domain=sample_domain)
    assert isinstance(score_mean, float), "score_mean is not a float"

    # Test softmax normalization
    scorer = MaNoScorer(threshold=float("inf"))
    score_mean = scorer(estimator, X, y, sample_domain=sample_domain)
    assert isinstance(score_mean, float), "score_mean is not a float"

    # Test invalid p-norm order
    with pytest.raises(ValueError):
        MaNoScorer(p=-1)

    # Test correct output range
    scorer = MaNoScorer()
    score_mean = scorer(estimator, X, y, sample_domain=sample_domain)
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
        .set_score_request(sample_weight=True)
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


@pytest.mark.parametrize("threshold", [-1.0, 5.0, float("inf")])
def test_mano_criterion_preserves_logits(threshold):
    """Criterion evaluation must not center the logits used by Taylor."""
    logits = np.array([[1.0, 2.0, 3.0], [2.0, -1.0, 0.0]])
    original = logits.copy()
    scorer = MaNoScorer(threshold=threshold)
    criterion = scorer._get_criterion(logits)
    np.testing.assert_array_equal(logits, original)
    expected_log_proba = original - np.log(np.exp(original).sum(axis=1, keepdims=True))
    np.testing.assert_allclose(criterion, -expected_log_proba.mean())
    actual = scorer._softrun(logits, criterion, threshold)
    if criterion > threshold:
        expected = np.exp(original) / np.exp(original).sum(axis=1, keepdims=True)
    else:
        expected = 1 + original + original**2 / 2
        expected -= expected.min(axis=1, keepdims=True)
        expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(logits, original)


def test_mano_read_only_logits():
    """A diagnostic criterion accepts immutable model outputs."""
    logits = np.array([[1.0, 2.0, 3.0]])
    logits.setflags(write=False)
    assert np.isfinite(MaNoScorer()._get_criterion(logits))


@pytest.mark.parametrize("row", [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0], [-2.0, 0.0]])
def test_mano_zero_taylor_row(row):
    """A zero shifted polynomial stays zero, like L1-normalize in the reference."""
    logits = np.array([row])
    with np.errstate(divide="raise", invalid="raise"):
        result = MaNoScorer._taylor_softmax(logits)
    np.testing.assert_array_equal(result, np.zeros_like(logits))


@pytest.mark.parametrize("constant_outputs", [False, True])
def test_mano_public_deep_score_uses_original_logits(constant_outputs):
    """Exercise actual Torch inference and compare the public scorer with Eq. 6."""
    torch = pytest.importorskip("torch")

    class DeepEstimator:
        def __init__(self):
            self.module_ = torch.nn.Linear(1, 3, dtype=torch.float64)
            self.module_.layer_name = "classifier"
            with torch.no_grad():
                self.module_.weight.copy_(torch.tensor([[1.0], [-1.0], [2.0]]))
                self.module_.bias.copy_(torch.tensor([1.0, 2.0, 3.0]))
                if constant_outputs:
                    self.module_.weight.zero_()
                    self.module_.bias.fill_(2.0)

        def infer(self, X):
            self.last_logits = self.module_(torch.as_tensor(X, dtype=torch.float64))
            return self.last_logits

        def predict_proba(self, X):
            return torch.softmax(self.infer(X), dim=-1).detach().numpy()

    X = np.array([[1.0], [0.0], [0.5], [-0.5]])
    y = np.array([0, -1, -1, -1])
    domain = np.array([1, -1, -1, -1])
    estimator = DeepEstimator()
    raw = estimator.infer(X[1:]).detach().numpy().copy()
    polynomial = 1 + torch.from_numpy(raw) + torch.from_numpy(raw).square() / 2
    polynomial = polynomial - polynomial.min(dim=1, keepdim=True).values
    expected = torch.nn.functional.normalize(polynomial, p=1, dim=1)
    expected_score = (
        torch.linalg.vector_norm(expected, ord=4) / expected.numel() ** 0.25
    ).item()
    scorer = MaNoScorer(threshold=float("inf"))
    actual = scorer(estimator, X, y, sample_domain=domain)
    np.testing.assert_allclose(actual, expected_score)
    np.testing.assert_array_equal(estimator.last_logits.detach().numpy(), raw)
