import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression

try:
    import torch
except ImportError:
    torch = False

import pytest

from skada import (
    JCPOTLabelProp,
    JCPOTLabelPropAdapter,
    OTLabelProp,
    OTLabelPropAdapter,
    make_da_pipeline,
    source_target_split,
)
from skada.datasets import DomainAwareDataset


@pytest.mark.parametrize(
    "estimator",
    [
        make_da_pipeline(OTLabelPropAdapter(), LogisticRegression()),
        make_da_pipeline(OTLabelPropAdapter(reg=10), LogisticRegression()),
        OTLabelProp(LogisticRegression()),
        OTLabelProp(),
        make_da_pipeline(JCPOTLabelPropAdapter(), LogisticRegression()),
        JCPOTLabelProp(),
    ],
)
def test_label_prop_estimator(estimator, da_blobs_dataset):
    X, y, sample_domain = da_blobs_dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=False
    )
    X_source, X_target, y_source, y_target = source_target_split(
        X, y, sample_domain=sample_domain
    )

    # Just scale some feature to avoid having an identity cov matrix
    X_scaled = np.copy(X_source)
    X_target_scaled = np.copy(X_target)
    X_scaled[:, 0] *= 2
    X_target_scaled[:, 1] *= 3
    dataset = DomainAwareDataset(
        [
            (X_scaled, y_source, "s"),
            (X_target_scaled, y_target, "t"),
        ]
    )

    X_train, y_train, sample_domain = dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )
    estimator.fit(X_train, y_train, sample_domain=sample_domain)
    X_test, y_test, sample_domain = dataset.pack(
        as_sources=[], as_targets=["t"], mask_target_labels=False
    )
    y_pred = estimator.predict(X_test, sample_domain=sample_domain)
    assert np.mean(y_pred == y_test) > 0.9
    score = estimator.score(X_test, y_test, sample_domain=sample_domain)
    assert score > 0.9


@pytest.mark.parametrize(
    "estimator",
    [
        make_da_pipeline(
            OTLabelPropAdapter(), KernelRidge().set_fit_request(sample_weight=True)
        ),
    ],
)
def test_label_prop_estimator_reg(estimator, da_reg_dataset):
    X, y, sample_domain = da_reg_dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=False
    )
    X_source, X_target, y_source, y_target = source_target_split(
        X, y, sample_domain=sample_domain
    )
    w = np.ones_like(y)

    sample_domain_test = sample_domain[sample_domain < 0]
    y_train = y.copy()
    y_train[sample_domain < 0] = -1

    estimator.fit(X, y_train, sample_domain=sample_domain)

    estimator.fit(X, y_train, sample_weight=w, sample_domain=sample_domain)

    y_pred = estimator.predict(X_target)
    assert np.mean((y_pred - y_target) ** 2) < 2
    score = estimator.score(X_target, y_target, sample_domain=sample_domain_test)
    assert score > -0.5
