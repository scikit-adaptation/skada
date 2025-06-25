# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#         Antoine Collas <contact@antoinecollas.fr>
#
# License: BSD 3-Clause

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.utils import check_random_state

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
    MultiLinearMongeAlignment,
    MultiLinearMongeAlignmentAdapter,
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
        make_da_pipeline(MultiLinearMongeAlignmentAdapter(), LogisticRegression()),
        MultiLinearMongeAlignment(),
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
        make_da_pipeline(OTMappingAdapter(), Ridge()),
        OTMapping(Ridge()),
        make_da_pipeline(EntropicOTMappingAdapter(), Ridge()),
        EntropicOTMapping(Ridge()),
        make_da_pipeline(LinearOTMappingAdapter(), Ridge()),
        LinearOTMapping(Ridge()),
        make_da_pipeline(CORALAdapter(), Ridge()),
        CORAL(Ridge(), assume_centered=False),
        CORAL(Ridge(), assume_centered=True),
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
def test_reg_mapping_estimator(estimator):
    dataset = make_shifted_datasets(
        n_samples_source=5,
        n_samples_target=10,
        shift="conditional_shift",
        mean=0.5,
        noise=0.3,
        label="regression",
        random_state=42,
        return_dataset=True,
    )
    X_train, y_train, sample_domain_train = dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )
    estimator.fit(X_train, y_train, sample_domain=sample_domain_train)
    X_test, y_test, sample_domain_test = dataset.pack(
        as_sources=[], as_targets=["t"], mask_target_labels=False
    )
    score = estimator.score(X_test, y_test, sample_domain=sample_domain_test)
    # xxx(okachaiev): take care of those test, this result is rather bad
    assert score >= -1.0  # Ridge uses R^2, so it can be < 0.


def _base_test_new_X_adapt(estimator, da_dataset):
    X_train, y_train, sample_domain = da_dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )
    true_X_adapt = estimator.fit_transform(
        X_train, y_train, sample_domain=sample_domain
    )

    # Adapt with new X, i.e. same domain, different samples
    rng = check_random_state(42)
    idx = rng.choice(len(X_train), len(X_train) // 5, replace=False)
    X_adapt = estimator.transform(
        X_train[idx] + 1e-8,
        y_train[idx],
        sample_domain=sample_domain[idx],
        allow_source=True,
    )

    # Check that the adapted data are the same
    assert np.allclose(true_X_adapt[idx], X_adapt)

    # Check it adapts even if some target classes are not present in the new X
    classes = np.unique(y_train)[::2]
    mask = np.isin(y_train, classes)
    X_train = X_train[mask]
    y_train = y_train[mask]
    sample_domain = sample_domain[mask]
    X_adapt = estimator.transform(
        X_train, y_train, sample_domain=sample_domain, allow_source=True
    )

    # Check that the adapted data are the same
    true_X_adapt = true_X_adapt[mask]
    assert np.allclose(true_X_adapt, X_adapt)


@pytest.mark.parametrize(
    "estimator",
    [
        (OTMappingAdapter()),
        (EntropicOTMappingAdapter()),
        (ClassRegularizerOTMappingAdapter(norm="lpl1")),
        (ClassRegularizerOTMappingAdapter(norm="l1l2")),
        (LinearOTMappingAdapter()),
        (CORALAdapter(assume_centered=True)),
        (
            pytest.param(
                MMDLSConSMappingAdapter(gamma=1e-3),
                marks=pytest.mark.skipif(not torch, reason="PyTorch not installed"),
            )
        ),
    ],
)
def test_new_X_adapt(estimator):
    # for dataset in da_reg_datasets:
    dataset = make_shifted_datasets(
        n_samples_source=5,
        n_samples_target=10,
        shift="conditional_shift",
        mean=0.5,
        noise=0.3,
        label="regression",
        random_state=42,
        return_dataset=True,
    )
    _base_test_new_X_adapt(estimator, dataset)


@pytest.mark.parametrize(
    "estimator",
    [
        OTMappingAdapter(),
        EntropicOTMappingAdapter(),
        LinearOTMappingAdapter(),
        CORALAdapter(assume_centered=True),
        pytest.param(
            MMDLSConSMappingAdapter(gamma=1e-3),
            marks=pytest.mark.skipif(not torch, reason="PyTorch not installed"),
        ),
    ],
)
def test_reg_new_X_adapt(estimator):
    dataset = make_shifted_datasets(
        n_samples_source=5,
        n_samples_target=10,
        shift="conditional_shift",
        mean=0.5,
        noise=0.3,
        label="regression",
        random_state=42,
        return_dataset=True,
    )
    _base_test_new_X_adapt(estimator, dataset)


@pytest.mark.parametrize(
    "estimator",
    [
        OTMapping(),
        EntropicOTMapping(),
        ClassRegularizerOTMapping(),
        ClassRegularizerOTMapping(norm="l1l2"),
        LinearOTMapping(),
        CORAL(),
        pytest.param(
            MMDLSConSMapping(),
            marks=pytest.mark.skipif(not torch, reason="PyTorch not installed"),
        ),
    ],
)
def test_mapping_source_samples(estimator, da_blobs_dataset):
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

    # Adapt with new X, i.e. same domain, different samples
    rng = check_random_state(42)
    idx = rng.choice(len(X_train), len(X_train) // 5, replace=False)

    y_pred = estimator.predict(
        X_train[idx], sample_domain=sample_domain[idx], allow_source=True
    )
    assert y_pred.shape[0] == len(idx)
