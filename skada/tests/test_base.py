# Author: Yanis Lalou <yanis.lalou@polytechnique.edu>
#
# License: BSD 3-Clause

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_regression

from skada import SubspaceAlignmentAdapter, make_da_pipeline
from skada.datasets import make_shifted_datasets
from skada.utils import extract_source_indices
from skada._utils import (
    _DEFAULT_SOURCE_DOMAIN_LABEL,
    _DEFAULT_TARGET_DOMAIN_LABEL
)

import pytest


def test_base_selector_remove_masked():
    n_samples = 10
    X, y, sample_domain = make_shifted_datasets(
        n_samples_source=n_samples,
        n_samples_target=n_samples,
        shift='concept_drift',
        noise=0.1,
        random_state=42,
    )

    pipe = make_da_pipeline(
        LogisticRegression(),
    )

    selector = pipe['logisticregression']
    X_output, y_output, routed_params = selector._remove_masked(
        X, y, selector.get_params()
    )

    assert X_output.shape[0] == 2 * n_samples * 8, "X output shape mismatch"
    assert y_output.shape[0] == 2 * n_samples * 8, "y output shape mismatch"

    source_idx = extract_source_indices(sample_domain)
    y[~source_idx] = -1  # We mask the target labels

    X_output, y_output, routed_params = selector._remove_masked(
        X, y, selector.get_params()
    )

    assert X_output.shape[0] != 2 * n_samples * 8, "X output shape mismatch"
    assert y_output.shape[0] != 2 * n_samples * 8, "y output shape mismatch"
    assert X_output.shape[0] == y_output.shape[0]


@pytest.mark.parametrize('step', [SubspaceAlignmentAdapter(), LogisticRegression()])
def test_base_selector_remove_masked_transform(step):
    # Test the remove masked method if the selector
    # has or has not a transform method
    # xxx(okachaiev): I'm not sure that the code actually tests that

    n_samples = 10
    X, y, sample_domain = make_shifted_datasets(
        n_samples_source=n_samples,
        n_samples_target=n_samples,
        shift='concept_drift',
        noise=0.1,
        random_state=42,
    )

    pipe = make_da_pipeline(step)
    # no ValueError is raised
    pipe.fit(X=X, y=y, sample_domain=sample_domain)


def test_base_selector_remove_masked_continuous():
    # Same as test_base_selector_remove_masked but with continuous labels
    n_samples = 10
    X, y = make_regression(
        n_samples=n_samples,
        n_features=2,
        noise=1,
        random_state=42
    )

    pipe = make_da_pipeline(
        LogisticRegression(),
    )

    selector = pipe['logisticregression']

    # We attribute a random source/target label to each sample
    source_idx = np.random.choice([False, True], size=len(y))
    y[~source_idx] = -1  # We mask the target labels

    X_output, y_output, routed_params = selector._remove_masked(
        X, y, selector.get_params()
    )

    assert X_output.shape[0] != X.shape[0], "X output shape mismatch"
    assert y_output.shape[0] != y.shape[0], "y output shape mismatch"
    assert X_output.shape[0] == y_output.shape[0]


def test_base_selector_domains():
    n_samples = 10
    X, y, sample_domain = make_shifted_datasets(
        n_samples_source=n_samples,
        n_samples_target=n_samples,
        shift='concept_drift',
        noise=0.1,
        random_state=42,
    )

    pipe = make_da_pipeline(
        LogisticRegression(),
    )

    pipe.fit(X, y)

    assert (pipe['logisticregression'].domains_ ==
            set([
                _DEFAULT_SOURCE_DOMAIN_LABEL,
                _DEFAULT_TARGET_DOMAIN_LABEL
            ])
            )

    # Create a new pipe to test with the sample_domain argument
    pipe = make_da_pipeline(
        LogisticRegression(),
    )

    pipe.fit(X, y, sample_domain=sample_domain)

    assert (pipe['logisticregression'].domains_ ==
            set(np.unique(sample_domain))
            )
