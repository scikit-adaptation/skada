# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#
# License: BSD 3-Clause

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from skada import (
    SubspaceAlignment,
    SubspaceAlignmentAdapter,
    TransferComponentAnalysis,
    TransferComponentAnalysisAdapter,
    TransferJointMatching,
    TransferJointMatchingAdapter,
    make_da_pipeline,
)
from skada.base import AdaptationOutput
from skada.datasets import DomainAwareDataset


@pytest.mark.parametrize(
    "estimator",
    [
        make_da_pipeline(
            SubspaceAlignmentAdapter(n_components=2), LogisticRegression()
        ),
        SubspaceAlignment(n_components=2),
        make_da_pipeline(
            TransferComponentAnalysisAdapter(n_components=2), LogisticRegression()
        ),
        TransferComponentAnalysis(n_components=2),
        TransferJointMatching(n_components=2, kernel="linear"),
        make_da_pipeline(
            TransferJointMatchingAdapter(n_components=2, kernel="linear", verbose=True),
            LogisticRegression(),
        ),
    ],
)
def test_subspace_alignment(estimator, da_dataset):
    X_train, y_train, sample_domain = da_dataset.pack_train(
        as_sources=["s"], as_targets=["t"]
    )
    estimator.fit(X_train, y_train, sample_domain=sample_domain)
    X_test, y_test, sample_domain = da_dataset.pack_test(as_targets=["t"])
    y_pred = estimator.predict(X_test, sample_domain=sample_domain)
    assert np.mean(y_pred == y_test) > 0.9
    score = estimator.score(X_test, y_test, sample_domain=sample_domain)
    assert score > 0.9


@pytest.mark.parametrize(
    "adapter, n_samples, n_features, n_components",
    [
        (SubspaceAlignmentAdapter(), 5, 3, 3),
        (SubspaceAlignmentAdapter(), 2, 5, 2),
        (TransferComponentAnalysisAdapter(), 5, 3, 3),
        (TransferComponentAnalysisAdapter(), 2, 3, 3),
        (TransferComponentAnalysisAdapter(), 2, 5, 4),
        (TransferJointMatchingAdapter(), 5, 3, 3),
        (TransferJointMatchingAdapter(n_components=3), 2, 3, 3),
        (TransferJointMatchingAdapter(n_components=4), 2, 5, 4),
    ],
)
def test_subspace_default_n_components(adapter, n_samples, n_features, n_components):
    rng = np.random.RandomState(42)
    X_source, y_source, X_target, y_target = (
        rng.rand(n_samples, n_features),
        np.eye(n_samples, dtype="int32")[0],
        rng.rand(n_samples, n_features),
        np.eye(n_samples, dtype="int32")[0],
    )
    dataset = DomainAwareDataset(
        [
            (X_source, y_source, "s"),
            (X_target, y_target, "t"),
        ]
    )

    X_train, y_train, sample_domain = dataset.pack_train(
        as_sources=["s"], as_targets=["t"]
    )
    adapter.fit(X_train, y_train, sample_domain=sample_domain)
    X_test, _, sample_domain = dataset.pack_test(as_targets=["t"])
    output = adapter.transform(X_test, sample_domain=sample_domain)
    if isinstance(output, AdaptationOutput):
        output = output.X
    assert output.shape[1] == n_components
