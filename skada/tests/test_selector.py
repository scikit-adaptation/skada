# Author: Yanis Lalou <yanis.lalou@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#
# License: BSD 3-Clause

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_regression

from skada import SubspaceAlignmentAdapter, make_da_pipeline
from skada.datasets import make_shifted_datasets
from skada.utils import extract_source_indices
from skada._utils import (
    _DEFAULT_MASKED_TARGET_CLASSIFICATION_LABEL,
    _DEFAULT_MASKED_TARGET_REGRESSION_LABEL,
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
    X_output, y_output, _ = selector._remove_masked(X, y, {})

    assert X_output.shape[0] == 2 * n_samples * 8, "X output shape mismatch"
    assert X_output.shape[0] == y_output.shape[0]

    source_idx = extract_source_indices(sample_domain)
    # mask target labels
    y[~source_idx] = _DEFAULT_MASKED_TARGET_CLASSIFICATION_LABEL
    X_output, y_output, _ = selector._remove_masked(X, y, {})

    assert X_output.shape[0] == n_samples * 8, "X output shape mismatch"
    assert X_output.shape[0] == y_output.shape[0]


@pytest.mark.parametrize('step', [SubspaceAlignmentAdapter(), LogisticRegression()])
def test_base_selector_remove_masked_transform(step):
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
    # Same as `test_base_selector_remove_masked` but with continuous labels
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

    # randomly designate each sample as source (True) or target (False)
    rng = np.random.default_rng(42)
    source_idx = rng.choice([False, True], size=n_samples)
    # mask target labels
    y[~source_idx] = _DEFAULT_MASKED_TARGET_REGRESSION_LABEL
    X_output, y_output, _ = selector._remove_masked(X, y, {})

    n_target_samples = X.shape[0] - np.sum(source_idx)
    assert X_output.shape[0] == n_target_samples, "X output shape mismatch"
    assert X_output.shape[0] == y_output.shape[0]
