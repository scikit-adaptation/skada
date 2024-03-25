# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#         Antoine Collas <contact@antoinecollas.fr>
#
# License: BSD 3-Clause

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC

try:
    import torch
except ImportError:
    torch = False

import pytest

from skada import (
    CORAL,
    ClassRegularizerOTMapping,
    ClassRegularizerOTMappingAdapter,
    CORALAdapter,
    EntropicOTMapping,
    EntropicOTMappingAdapter,
    LinearOTMapping,
    LinearOTMappingAdapter,
    MMDLSConSMapping,
    MMDLSConSMappingAdapter,
    OTMapping,
    OTMappingAdapter,
    make_da_pipeline,
    source_target_split,
)
from skada.datasets import DomainAwareDataset, make_shifted_datasets


@pytest.mark.parametrize(
    "estimator",
    [
        make_da_pipeline(OTMappingAdapter(), LogisticRegression()),
        OTMapping(),
        make_da_pipeline(EntropicOTMappingAdapter(), LogisticRegression()),
        EntropicOTMapping(),
        make_da_pipeline(
            ClassRegularizerOTMappingAdapter(norm="lpl1"), LogisticRegression()
        ),
        ClassRegularizerOTMapping(),
        make_da_pipeline(
            ClassRegularizerOTMappingAdapter(norm="l1l2"), LogisticRegression()
        ),
        ClassRegularizerOTMapping(norm="l1l2"),
        make_da_pipeline(LinearOTMappingAdapter(), LogisticRegression()),
        LinearOTMapping(),
        make_da_pipeline(CORALAdapter(), LogisticRegression()),
        pytest.param(
            CORALAdapter(reg=None),
            marks=pytest.mark.xfail(reason="Fails without regularization"),
        ),
        make_da_pipeline(CORALAdapter(reg=0.1), LogisticRegression()),
        CORAL(),
        pytest.param(
            make_da_pipeline(MMDLSConSMappingAdapter(gamma=1e-3), SVC()),
            marks=pytest.mark.skipif(not torch, reason="PyTorch not installed"),
        ),
        pytest.param(
            MMDLSConSMapping(),
            marks=pytest.mark.skipif(not torch, reason="PyTorch not installed"),
        ),
    ],
)
def test_mapping_estimator(estimator, da_blobs_dataset):
    X, y, sample_domain = da_blobs_dataset
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

    X_train, y_train, sample_domain = dataset.pack_train(
        as_sources=["s"], as_targets=["t"]
    )
    estimator.fit(X_train, y_train, sample_domain=sample_domain)
    X_test, y_test, sample_domain = dataset.pack_test(as_targets=["t"])
    y_pred = estimator.predict(X_test, sample_domain=sample_domain)
    assert np.mean(y_pred == y_test) > 0.9
    score = estimator.score(X_test, y_test, sample_domain=sample_domain)
    assert score > 0.9


@pytest.mark.parametrize(
    "estimator",
    [
        make_da_pipeline(OTMappingAdapter(), Ridge()),
        OTMapping(Ridge()),
        make_da_pipeline(EntropicOTMappingAdapter(), Ridge()),
        EntropicOTMapping(Ridge()),
        make_da_pipeline(LinearOTMappingAdapter(), Ridge()),
        LinearOTMapping(Ridge()),
        make_da_pipeline(CORALAdapter(), Ridge()),
        CORAL(Ridge()),
        pytest.param(
            make_da_pipeline(MMDLSConSMappingAdapter(gamma=1e-3), Ridge()),
            marks=pytest.mark.skipif(not torch, reason="PyTorch not installed"),
        ),
        pytest.param(
            MMDLSConSMapping(Ridge()),
            marks=pytest.mark.skipif(not torch, reason="PyTorch not installed"),
        ),
    ],
)
def test_reg_mapping_estimator(estimator, da_reg_dataset):
    X, y, sample_domain = da_reg_dataset
    Xs, Xt, ys, yt = source_target_split(X, y, sample_domain=sample_domain)
    estimator.fit(X, y, sample_domain=sample_domain)
    score = estimator.score(Xt, yt)
    assert score >= 0


def _base_test_new_X_adapt(estimator, da_dataset):
    X_train, y_train, sample_domain = da_dataset

    estimator.fit(X_train, y_train, sample_domain=sample_domain)
    true_X_adapt = estimator.adapt(X_train, y_train, sample_domain=sample_domain)

    idx = np.random.choice(len(X_train), len(X_train) // 5, replace=False)

    # Adapt with new X, i.e. same domain, different samples
    X_adapt = estimator.adapt(
        X_train[idx] + 1e-8, y_train[idx], sample_domain=sample_domain[idx]
    )

    # Check that the adapted data are the same
    assert np.allclose(true_X_adapt[idx], X_adapt)

    # Check it adapts even if some target classes are not present in the new X
    classes = np.unique(y_train)[::2]
    mask = np.isin(y_train, classes)
    X_train = X_train[mask]
    y_train = y_train[mask]
    sample_domain = sample_domain[mask]
    X_adapt = estimator.adapt(X_train, y_train, sample_domain=sample_domain)

    # Check that the adapted data are the same
    true_X_adapt = true_X_adapt[mask]
    assert np.allclose(true_X_adapt, X_adapt)


@pytest.mark.parametrize(
    "estimator",
    [
        OTMappingAdapter(),
        EntropicOTMappingAdapter(),
        ClassRegularizerOTMappingAdapter(norm="lpl1"),
        ClassRegularizerOTMappingAdapter(norm="l1l2"),
        LinearOTMappingAdapter(),
        CORALAdapter(),
        pytest.param(
            MMDLSConSMappingAdapter(gamma=1e-3),
            marks=pytest.mark.skipif(not torch, reason="PyTorch not installed"),
        ),
    ],
)
def test_new_X_adapt(estimator, da_dataset):
    da_dataset = da_dataset.pack_train(as_sources=["s"], as_targets=["t"])

    _base_test_new_X_adapt(estimator, da_dataset)


@pytest.mark.parametrize(
    "estimator",
    [
        OTMappingAdapter(),
        EntropicOTMappingAdapter(),
        LinearOTMappingAdapter(),
        CORALAdapter(),
        pytest.param(
            MMDLSConSMappingAdapter(gamma=1e-3),
            marks=pytest.mark.skipif(not torch, reason="PyTorch not installed"),
        ),
    ],
)
def test_reg_new_X_adapt(estimator, da_reg_dataset):
    _base_test_new_X_adapt(estimator, da_reg_dataset)


@pytest.mark.parametrize(
    "estimator",
    [
        OTMappingAdapter(),
        EntropicOTMappingAdapter(),
        ClassRegularizerOTMappingAdapter(norm="lpl1"),
        ClassRegularizerOTMappingAdapter(norm="l1l2"),
        LinearOTMappingAdapter(),
        CORALAdapter(),
        pytest.param(
            MMDLSConSMappingAdapter(gamma=1e-3),
            marks=pytest.mark.skipif(not torch, reason="PyTorch not installed"),
        ),
    ],
)
def test_X_source_target_different_size(estimator):
    # Test when X_source.shape[0] > X_target.shape[0]
    X, y, sample_domain = make_shifted_datasets(
        n_samples_source=20,
        n_samples_target=10,
        shift="covariate_shift",
        noise=None,
        return_dataset=False,
    )

    X_source, X_target, y_source, y_target = source_target_split(
        X, y, sample_domain=sample_domain
    )
    assert X_source.shape[0] > X_target.shape[0]

    # No exception should be raised
    estimator.fit(X, y, sample_domain=sample_domain)

    # Test when X_source.shape[0] < X_target.shape[0]
    X, y, sample_domain = make_shifted_datasets(
        n_samples_source=10,
        n_samples_target=20,
        shift="covariate_shift",
        noise=None,
        return_dataset=False,
    )

    X_source, X_target, y_source, y_target = source_target_split(
        X, y, sample_domain=sample_domain
    )
    assert X_source.shape[0] < X_target.shape[0]

    # No exception should be raised
    estimator.fit(X, y, sample_domain=sample_domain)
