# Author: Yanis Lalou <yanis.lalou@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#
# License: BSD 3-Clause

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.datasets import make_regression
from sklearn.linear_model import LogisticRegression
from sklearn.utils.metadata_routing import get_routing_for_object

from skada import SubspaceAlignmentAdapter, make_da_pipeline
from skada._utils import (
    _DEFAULT_MASKED_TARGET_CLASSIFICATION_LABEL,
    _DEFAULT_MASKED_TARGET_REGRESSION_LABEL,
    _remove_masked,
)
from skada.base import (
    AdaptationOutput,
    IncompatibleMetadataError,
    PerDomain,
    SelectSource,
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
        shift="concept_drift",
        noise=0.1,
        random_state=42,
    )

    lr = LogisticRegression()
    pipe = make_da_pipeline(lr)
    selector = pipe[0]

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
        shift="concept_drift",
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
        shift="concept_drift",
        noise=0.1,
        random_state=42,
        return_dataset=True,
    )
    X, y, sample_domain = dataset.pack_train(as_sources=["s"], as_targets=["t"])

    pipe = make_da_pipeline(step)
    # no ValueError is raised
    pipe.fit(X=X, y=y, sample_domain=sample_domain)


def test_base_selector_no_filtering_transformer():
    dataset = make_shifted_datasets(
        n_samples_source=10,
        n_samples_target=20,
        shift="concept_drift",
        noise=0.1,
        random_state=42,
        return_dataset=True,
    )
    X_train, y_train, sample_domain = dataset.pack_train(
        as_sources=["s"], as_targets=["t"]
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


@pytest.mark.parametrize("estimator_cls", [PerDomain, Shared])
def test_selector_inherits_routing(estimator_cls):
    lr = LogisticRegression().set_fit_request(sample_weight=True)
    estimator = estimator_cls(lr)
    routing = get_routing_for_object(estimator)
    assert "sample_weight" in routing.consumes("fit", ["sample_weight"])


def test_selector_rejects_incompatible_adaptation_output():
    X = AdaptationOutput(np.ones((10, 2)), sample_weight=np.zeros(10))
    y = np.zeros(10)
    estimator = Shared(LogisticRegression())

    with pytest.raises(IncompatibleMetadataError):
        estimator.fit(X, y)


@pytest.mark.parametrize(
    "selector_cls, side",
    [
        (SelectSource, "source"),
        (SelectTarget, "target"),
    ],
)
def test_source_selector(da_multiclass_dataset, selector_cls, side):
    X, y, sample_domain = da_multiclass_dataset
    X_source, X_target = source_target_split(X, sample_domain=sample_domain)
    output = {}

    class FakeEstimator(BaseEstimator):
        def fit(self, X, y):
            output["n_X_samples"] = X.shape[0]
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

    # should allow everything for predict
    pipe.predict(X)
    assert output["n_predict_samples"] == X.shape[0]
