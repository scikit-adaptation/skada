# Authors: Julie Alberge and Félix Lefebvre
#
# License: BSD 3-Clause

import pytest
from sklearn.preprocessing import StandardScaler

from skada._gradual_da import GradualEstimator
from skada._pipeline import make_da_pipeline
from skada.datasets import make_shifted_datasets
from skada.utils import check_X_y_domain


@pytest.mark.parametrize(
    "label, n, m, advanced_ot_plan_sampling",
    [
        ("binary", 15, 5, False),
        ("binary", 10, 10, False),
        ("binary", 5, 15, False),
        ("binary", 15, 5, True),
        ("binary", 10, 10, True),
        ("binary", 5, 15, True),
    ],
)
def test_gradual_estimator(label, n, m, advanced_ot_plan_sampling):
    X, y, sample_domain = make_shifted_datasets(
        n_samples_source=n,
        n_samples_target=m,
        shift="covariate_shift",
        noise=None,
        label=label,
    )
    X, y, sample_domain = check_X_y_domain(X, y, sample_domain)

    clf_gradual = GradualEstimator(
        T=5,
        advanced_ot_plan_sampling=advanced_ot_plan_sampling,
        save_estimators=True,
    ).fit(X, y, sample_domain=sample_domain)

    assert (
        clf_gradual.predict(X).shape == y.shape
    ), "Wrong shape of the predicted y-values (labels) when using `predict` method"

    if label == "binary":
        assert clf_gradual.predict_proba(X).shape == (
            y.shape[0],
            2,
        ), "Wrong shape of the output when using `predict_proba` method"

        assert clf_gradual.predict_log_proba(X).shape == (
            y.shape[0],
            2,
        ), "Wrong shape of the output when using `predict_log_proba` method"

    assert clf_gradual.score(X, y) >= 0, "The score should be non-negative"

    # Test get_intermediate_estimators
    intermediate_estimators = clf_gradual.get_intermediate_estimators()
    assert isinstance(intermediate_estimators, list)
    # T=5, so there should be 5 intermediate estimators + the final one
    assert len(intermediate_estimators) == 6

    # The `GradualEstimator` should be usable with `make_da_pipeline`
    manage_pipeline = False
    try:
        clf_gradual = make_da_pipeline(StandardScaler(), GradualEstimator(T=5)).fit(
            X, y, sample_domain=sample_domain
        )
        manage_pipeline = True
    finally:
        assert manage_pipeline, "Couldn't use make_da_pipeline with GradualEstimator"

    # The `GradualEstimator` should accept sample_domain as an argument
    if manage_pipeline:
        clf_gradual.fit(X, y, sample_domain=sample_domain)
        clf_gradual.predict(X, sample_domain=sample_domain)
