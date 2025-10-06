# Author: Yanis Lalou <yanis.lalou@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#
# License: BSD 3-Clause

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.datasets import make_regression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.metadata_routing import get_routing_for_object
from sklearn.utils.metaestimators import available_if

from skada import SubspaceAlignmentAdapter, make_da_pipeline
from skada._utils import (
    _DEFAULT_MASKED_TARGET_CLASSIFICATION_LABEL,
    _DEFAULT_MASKED_TARGET_REGRESSION_LABEL,
    _remove_masked,
)
from skada.base import (
    BaseAdapter,
    IncompatibleMetadataError,
    PerDomain,
    SelectSource,
    SelectSourceTarget,
    SelectTarget,
    Shared,
)
from skada.datasets import make_shifted_datasets
from skada.utils import extract_source_indices, source_target_split


def test_base_selector_estimator_fetcher():
    n_samples = 10
    X, y, sample_domain = make_shifted_datasets(
        n_samples_source=n_samples,
        n_samples_target=n_samples,
        shift="conditional_shift",
        noise=0.1,
        random_state=42,
    )

    lr = LogisticRegression()
    pipe = make_da_pipeline(lr)
    selector = pipe[-1]

    # before fitting, raises
    with pytest.raises(ValueError):
        selector.get_estimator()

    # after fitting, gives fitted estimator
    pipe.fit(X, y, sample_domain=sample_domain)
    assert isinstance(selector.get_estimator(), LogisticRegression)


def test_remove_masked_helper():
    n_samples = 10
    X, y, sample_domain = make_shifted_datasets(
        n_samples_source=n_samples,
        n_samples_target=n_samples,
        shift="conditional_shift",
        noise=0.1,
        random_state=42,
    )

    X_output, y_output, _ = _remove_masked(X, y, {})

    assert X_output.shape[0] == 2 * n_samples * 8, "X output shape mismatch"
    assert X_output.shape[0] == y_output.shape[0]

    source_idx = extract_source_indices(sample_domain)
    # mask target labels
    y[~source_idx] = _DEFAULT_MASKED_TARGET_CLASSIFICATION_LABEL
    X_output, y_output, _ = _remove_masked(X, y, {})

    assert X_output.shape[0] == n_samples * 8, "X output shape mismatch"
    assert X_output.shape[0] == y_output.shape[0]


@pytest.mark.parametrize("step", [SubspaceAlignmentAdapter(), LogisticRegression()])
def test_base_selector_remove_masked(step):
    n_samples = 10
    dataset = make_shifted_datasets(
        n_samples_source=n_samples,
        n_samples_target=n_samples,
        shift="conditional_shift",
        noise=0.1,
        random_state=42,
        return_dataset=True,
    )
    X, y, sample_domain = dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )

    pipe = make_da_pipeline(step)
    # no ValueError is raised
    pipe.fit(X=X, y=y, sample_domain=sample_domain)


def test_base_selector_no_filtering_transformer():
    dataset = make_shifted_datasets(
        n_samples_source=10,
        n_samples_target=20,
        shift="conditional_shift",
        noise=0.1,
        random_state=42,
        return_dataset=True,
    )
    X_train, y_train, sample_domain = dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )

    output = {}

    class FakeTransformer(BaseEstimator):
        def fit(self, X, y=None):
            output["n_samples"] = X.shape[0]
            self.fitted_ = True

        def transform(self, X):
            return X

    pipe = make_da_pipeline(FakeTransformer())
    pipe.fit(X=X_train, y=y_train, sample_domain=sample_domain)

    assert output["n_samples"] == X_train.shape[0]


def test_base_selector_remove_masked_continuous():
    # Same as `test_base_selector_remove_masked` but with continuous labels
    n_samples = 10
    X, y = make_regression(n_samples=n_samples, n_features=2, noise=1, random_state=42)

    # randomly designate each sample as source (True) or target (False)
    rng = np.random.default_rng(42)
    source_idx = rng.choice([False, True], size=n_samples)
    # mask target labels
    y[~source_idx] = _DEFAULT_MASKED_TARGET_REGRESSION_LABEL
    assert np.any(~np.isfinite(y)), "at least one label is masked"

    X_output, y_output, _ = _remove_masked(X, y, {})
    assert np.all(np.isfinite(y_output)), "masks are removed"

    n_source_samples = np.sum(source_idx)
    assert X_output.shape[0] == n_source_samples, "X output shape mismatch"
    assert X_output.shape[0] == y_output.shape[0]


@pytest.mark.parametrize(
    "estimator_cls", [PerDomain, Shared, SelectSource, SelectTarget]
)
def test_selector_inherits_routing(estimator_cls):
    lr = LogisticRegression().set_fit_request(sample_weight=True)
    estimator = estimator_cls(lr)
    routing = get_routing_for_object(estimator)
    assert "sample_weight" in routing.consumes("fit", ["sample_weight"])


def test_selector_rejects_incompatible_adaptation_output():
    n_samples = 10
    X = np.ones((n_samples, 2))
    y = np.zeros(n_samples, dtype=np.int32)
    y[:5] = 1

    class FakeAdapter(BaseAdapter):
        def fit_transform(self, X, y=None, sample_domain=None, **params):
            return X, dict(sample_weight=np.ones(X.shape[0]))

        def transform(
            self, X, y=None, sample_domain=None, allow_source=False, **params
        ):
            return X

    # fails if this is an estimator (not transformer)
    estimator = make_da_pipeline(FakeAdapter(), LogisticRegression())
    with pytest.raises(IncompatibleMetadataError):
        estimator.fit(X, y)

    # fails if this is the last estimator transformer
    estimator = make_da_pipeline(FakeAdapter(), StandardScaler())
    with pytest.raises(IncompatibleMetadataError):
        estimator.fit(X, y)

    # does not fail for non-final transformer
    estimator = make_da_pipeline(
        FakeAdapter(), StandardScaler(), SVC().set_fit_request(sample_weight=True)
    )
    estimator.fit(X, y)


@pytest.mark.parametrize(
    "selector_cls, side",
    [
        (SelectSource, "source"),
        (SelectTarget, "target"),
    ],
)
def test_source_selector_with_estimator(da_multiclass_dataset, selector_cls, side):
    X, y, sample_domain = da_multiclass_dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=False
    )
    X_source, X_target = source_target_split(X, sample_domain=sample_domain)
    output = {}

    class FakeEstimator(BaseEstimator):
        def fit(self, X, y=None):
            output["n_X_samples"] = X.shape[0]
            if y is not None:
                output["n_y_samples"] = y.shape[0]
            self.fitted_ = True

        def predict(self, X):
            output["n_predict_samples"] = X.shape[0]
            return X

    pipe = make_da_pipeline(selector_cls(FakeEstimator()))
    pipe.fit(X, y, sample_domain=sample_domain)

    # make sure both X and y are filtered out
    correct_n_samples = (X_source if side == "source" else X_target).shape[0]
    assert output["n_X_samples"] == correct_n_samples
    assert output["n_y_samples"] == correct_n_samples

    # make sure y=None works as well
    pipe.fit(X, None, sample_domain=sample_domain)
    assert output["n_X_samples"] == correct_n_samples

    # should allow everything for predict
    pipe.predict(X)
    assert output["n_predict_samples"] == X.shape[0]


@pytest.mark.parametrize(
    "selector_cls, side, _fit_transform",
    [
        (SelectSource, "source", False),
        (SelectTarget, "target", False),
        (SelectSource, "source", True),
        (SelectTarget, "target", True),
    ],
)
def test_source_selector_with_transformer(
    da_multiclass_dataset, selector_cls, side, _fit_transform
):
    X, y, sample_domain = da_multiclass_dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=False
    )
    X_source, X_target = source_target_split(X, sample_domain=sample_domain)
    output = {}

    class FakeTransformer(BaseEstimator):
        def fit(self, X, y=None):
            output["n_X_samples"] = X.shape[0]
            self.fitted_ = True

        def transform(self, X):
            output["n_transform_samples"] = X.shape[0]
            return X

        @available_if(lambda _: _fit_transform)
        def fit_transform(self, X, y=None):
            self.fit(X)
            return self.transform(X)

    pipe = make_da_pipeline(selector_cls(FakeTransformer()))
    pipe.fit(X, sample_domain=sample_domain)

    # make sure both X samples are filtered out
    correct_n_samples = (X_source if side == "source" else X_target).shape[0]
    assert output["n_X_samples"] == correct_n_samples

    # make sure fit_transform gives the same result
    pipe.fit_transform(X, None, sample_domain=sample_domain)
    assert output["n_X_samples"] == correct_n_samples

    # should allow everything for transform
    pipe.transform(X)
    assert output["n_transform_samples"] == X.shape[0]


@pytest.mark.parametrize(
    "selector_cls, side",
    [
        (SelectSource, "source"),
        (SelectTarget, "target"),
    ],
)
def test_source_selector_with_weights(da_multiclass_dataset, selector_cls, side):
    X, y, sample_domain = da_multiclass_dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=False
    )
    sample_weight = np.ones(X.shape[0])
    X_source, X_target = source_target_split(X, sample_domain=sample_domain)
    output = {}

    class FakeEstimator(BaseEstimator):
        __metadata_request__fit = {"sample_weight": True}
        __metadata_request__predict = {"sample_weight": True}

        def fit(self, X, y=None, sample_weight=None):
            output["n_sample_weight"] = sample_weight.shape[0]
            self.fitted_ = True

        def predict(self, X, sample_weight=None):
            output["n_predict_sample_weight"] = sample_weight.shape[0]
            return X

    pipe = make_da_pipeline(selector_cls(FakeEstimator()))
    pipe.fit(X, y, sample_weight=sample_weight, sample_domain=sample_domain)

    # make sure sample_weight is properly filtered out
    correct_n_samples = (X_source if side == "source" else X_target).shape[0]
    assert output["n_sample_weight"] == correct_n_samples

    # should allow everything for predict
    pipe.predict(X, sample_weight=sample_weight)
    assert output["n_predict_sample_weight"] == X.shape[0]


@pytest.mark.parametrize(
    "source_estimator, target_estimator",
    [(StandardScaler(), None), (StandardScaler(), StandardScaler())],
)
def test_source_target_selector(
    da_multiclass_dataset, source_estimator, target_estimator
):
    X, y, sample_domain = da_multiclass_dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=False
    )
    source_masks = extract_source_indices(sample_domain)
    # make sure sources and targets have significantly different mean
    X[source_masks] += 100 * np.ones((source_masks.sum(), X.shape[1]))

    pipe = make_da_pipeline(
        SelectSourceTarget(source_estimator, target_estimator),
        SVC(),
    )

    # no errors should be raised
    pipe.fit(X, y, sample_domain=sample_domain)

    # no error is raised when only a single domain type is present
    pipe.predict(X[~source_masks], sample_domain=sample_domain[~source_masks])

    # make sure that scalers were trained on different inputs
    correct_mean = np.zeros(X.shape[1])
    source_estimator = pipe[0].get_estimator("source")
    assert np.allclose(
        source_estimator.transform(X[source_masks]).mean(0), correct_mean
    )
    assert not np.allclose(
        source_estimator.transform(X[~source_masks]).mean(0), correct_mean
    )

    target_estimator = pipe[0].get_estimator("target")
    assert not np.allclose(
        target_estimator.transform(X[source_masks]).mean(0), correct_mean
    )
    assert np.allclose(
        target_estimator.transform(X[~source_masks]).mean(0), correct_mean
    )


def test_source_target_selector_fails_on_missing_domain(da_multiclass_dataset):
    X, y, sample_domain = da_multiclass_dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=False
    )
    source_masks = extract_source_indices(sample_domain)
    pipe = make_da_pipeline(SelectSourceTarget(StandardScaler()), SVC())

    # fails without targets
    with pytest.raises(ValueError):
        pipe.fit(
            X[source_masks], y[source_masks], sample_domain=sample_domain[source_masks]
        )

    # fails without sources
    with pytest.raises(ValueError):
        pipe.fit(
            X[~source_masks],
            y[~source_masks],
            sample_domain=sample_domain[~source_masks],
        )


def test_source_target_selector_non_transformers():
    with pytest.raises(TypeError):
        SelectSourceTarget(StandardScaler(), SVC())


def test_select_target_raises_error_on_masking():
    """
    Check that SelectTarget raises a ValueError
    when mask_target_labels is True.
    """
    with pytest.raises(
        ValueError, match="Target labels cannot be masked for SelectTarget."
    ):
        SelectTarget(LogisticRegression(), mask_target_labels=True)


def test_select_source_target_raises_error_on_masking():
    """
    Check that SelectSourceTarget raises a ValueError
    when mask_target_labels is True.
    """
    with pytest.raises(
        ValueError, match="Target labels cannot be masked for SelectSourceTarget."
    ):
        SelectSourceTarget(LogisticRegression(), mask_target_labels=True)


def test_make_da_pipeline_with_select_source_target():
    """
    Check that make_da_pipeline can be instantiated
    with SelectSourceTarget.
    """
    make_da_pipeline(
        StandardScaler(),
        SelectSource(SVC()),
        default_selector=SelectSourceTarget,
        mask_target_labels=False,
    )
