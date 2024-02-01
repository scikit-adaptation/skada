# Author: Yanis Lalou <yanis.lalou@polytechnique.edu>
#
# License: BSD 3-Clause

from sklearn.decomposition import PCA

from skada import SubspaceAlignmentAdapter, make_da_pipeline
from skada.datasets import (
    make_shifted_datasets
)

from skada.utils import extract_source_indices


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
        PCA(n_components=2),
        SubspaceAlignmentAdapter(n_components=2)
    )

    selector = pipe['subspacealignmentadapter']
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
