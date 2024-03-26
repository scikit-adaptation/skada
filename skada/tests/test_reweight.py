# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#         Antoine Collas <contact@antoinecollas.fr>
#
# License: BSD 3-Clause

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge

from skada import (
    DensityReweight,
    DensityReweightAdapter,
    DiscriminatorReweight,
    DiscriminatorReweightAdapter,
    GaussianReweight,
    GaussianReweightAdapter,
    KLIEPReweight,
    KLIEPReweightAdapter,
    KMMReweight,
    KMMReweightAdapter,
    MMDTarSReweight,
    MMDTarSReweightAdapter,
    NearestNeighborReweight,
    NearestNeighborReweightAdapter,
    make_da_pipeline,
    source_target_split,
)
from skada.datasets import make_shifted_datasets


@pytest.mark.parametrize(
    "estimator",
    [
        make_da_pipeline(
            DensityReweightAdapter(),
            LogisticRegression().set_fit_request(sample_weight=True),
        ),
        DensityReweight(),
        make_da_pipeline(
            GaussianReweightAdapter(),
            LogisticRegression().set_fit_request(sample_weight=True),
        ),
        GaussianReweight(),
        make_da_pipeline(
            DiscriminatorReweightAdapter(),
            LogisticRegression().set_fit_request(sample_weight=True),
        ),
        DiscriminatorReweight(),
        make_da_pipeline(
            KLIEPReweightAdapter(gamma=[0.1, 1], random_state=42),
            LogisticRegression().set_fit_request(sample_weight=True),
        ),
        KLIEPReweight(gamma=[0.1, 1], random_state=42),
        KLIEPReweight(gamma=0.2),
        NearestNeighborReweight(
            LogisticRegression().set_fit_request(sample_weight=True),
            laplace_smoothing=True,
        ),
        NearestNeighborReweight(laplace_smoothing=True),
        make_da_pipeline(
            NearestNeighborReweightAdapter(laplace_smoothing=True),
            LogisticRegression().set_fit_request(sample_weight=True),
        ),
        make_da_pipeline(
            KMMReweightAdapter(gamma=0.1),
            LogisticRegression().set_fit_request(sample_weight=True),
        ),
        KMMReweight(),
        KMMReweight(eps=0.1),
        make_da_pipeline(
            MMDTarSReweightAdapter(gamma=1.0),
            LogisticRegression().set_fit_request(sample_weight=True),
        ),
        MMDTarSReweight(gamma=1.0),
    ],
)
def test_reweight_estimator(estimator, da_dataset):
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
    "estimator",
    [
        make_da_pipeline(
            DensityReweightAdapter(), Ridge().set_fit_request(sample_weight=True)
        ),
        DensityReweight(Ridge().set_fit_request(sample_weight=True)),
        make_da_pipeline(
            GaussianReweightAdapter(),
            Ridge().set_fit_request(sample_weight=True),
        ),
        GaussianReweight(Ridge().set_fit_request(sample_weight=True)),
        make_da_pipeline(
            DiscriminatorReweightAdapter(),
            Ridge().set_fit_request(sample_weight=True),
        ),
        DiscriminatorReweight(Ridge().set_fit_request(sample_weight=True)),
        make_da_pipeline(
            KLIEPReweightAdapter(gamma=[0.1, 1], random_state=42),
            Ridge().set_fit_request(sample_weight=True),
        ),
        KLIEPReweight(
            Ridge().set_fit_request(sample_weight=True), gamma=[0.1, 1], random_state=42
        ),
        KLIEPReweight(Ridge().set_fit_request(sample_weight=True), gamma=0.2),
        make_da_pipeline(
            KMMReweightAdapter(gamma=0.1), Ridge().set_fit_request(sample_weight=True)
        ),
        KMMReweight(Ridge().set_fit_request(sample_weight=True)),
        KMMReweight(Ridge().set_fit_request(sample_weight=True), eps=0.1),
        make_da_pipeline(
            MMDTarSReweightAdapter(gamma=1.0),
            Ridge().set_fit_request(sample_weight=True),
        ),
        MMDTarSReweight(Ridge().set_fit_request(sample_weight=True), gamma=1.0),
    ],
)
def test_reg_reweight_estimator(estimator, da_reg_dataset):
    X, y, sample_domain = da_reg_dataset
    Xs, Xt, ys, yt = source_target_split(X, y, sample_domain=sample_domain)
    estimator.fit(X, y, sample_domain=sample_domain)
    score = estimator.score(Xt, yt)
    assert score >= 0


def _base_test_new_X_adapt(estimator, da_dataset):
    X_train, y_train, sample_domain = da_dataset

    estimator.fit(X_train, y_train, sample_domain=sample_domain)
    res1 = estimator.adapt(X_train, y_train, sample_domain=sample_domain)
    idx = np.random.choice(X_train.shape[0], 10)
    true_weights = res1["sample_weight"][idx]

    # Adapt with new X, i.e. same domain, different samples
    res2 = estimator.adapt(
        X_train[idx, :] + 1e-8, y_train[idx], sample_domain=sample_domain[idx]
    )

    # Check that the normalized weights are the same
    true_weights = true_weights / np.sum(true_weights)
    res2["sample_weight"] = res2["sample_weight"] / np.sum(res2["sample_weight"])
    assert np.allclose(true_weights, res2["sample_weight"])

    # Check it adapts even if some target classes are not present in the new X
    classes = np.unique(y_train)[::2]
    mask = np.isin(y_train, classes)
    X_train = X_train[mask]
    y_train = y_train[mask]
    sample_domain = sample_domain[mask]
    res3 = estimator.adapt(X_train, y_train, sample_domain=sample_domain)

    # Check that the normalized weights are the same
    true_weights = res1["sample_weight"][mask]
    true_weights = true_weights / np.sum(true_weights)
    res3["sample_weight"] = res3["sample_weight"] / np.sum(res3["sample_weight"])
    assert np.allclose(true_weights, res3["sample_weight"])


@pytest.mark.parametrize(
    "estimator, n_samples_source, n_samples_target",
    [
        (DensityReweightAdapter(), 5, 10),
        (DensityReweightAdapter(), 10, 5),
        (GaussianReweightAdapter(), 5, 10),
        (GaussianReweightAdapter(), 10, 5),
        (DiscriminatorReweightAdapter(), 5, 10),
        (DiscriminatorReweightAdapter(), 10, 5),
        (KLIEPReweightAdapter(gamma=[0.1, 1], random_state=42), 5, 10),
        (KLIEPReweightAdapter(gamma=[0.1, 1], random_state=42), 10, 5),
        (KMMReweightAdapter(gamma=0.1, smooth_weights=True), 5, 10),
        (KMMReweightAdapter(gamma=0.1, smooth_weights=True), 10, 5),
        (MMDTarSReweightAdapter(gamma=1.0), 5, 10),
        (MMDTarSReweightAdapter(gamma=1.0), 10, 5),
    ],
)
def test_new_X_adapt(estimator, n_samples_source, n_samples_target):
    da_dataset = make_shifted_datasets(
        n_samples_source=n_samples_source,
        n_samples_target=n_samples_target,
        shift="concept_drift",
        mean=0.5,
        noise=0.3,
        label="regression",
        random_state=42,
    )

    _base_test_new_X_adapt(estimator, da_dataset)


@pytest.mark.parametrize(
    "estimator",
    [
        DensityReweightAdapter(),
        GaussianReweightAdapter(),
        DiscriminatorReweightAdapter(),
        KLIEPReweightAdapter(gamma=[0.1, 1], random_state=42),
        KMMReweightAdapter(gamma=0.1, smooth_weights=True),
        MMDTarSReweightAdapter(gamma=1.0),
    ],
)
def test_reg_new_X_adapt(estimator, da_reg_dataset):
    _base_test_new_X_adapt(estimator, da_reg_dataset)


def test_reweight_warning(da_dataset):
    X_train, y_train, sample_domain = da_dataset.pack_train(
        as_sources=["s"], as_targets=["t"]
    )
    estimator = KLIEPReweightAdapter(gamma=0.1, max_iter=0)
    estimator.fit(X_train, y_train, sample_domain=sample_domain)

    with pytest.warns(
        UserWarning, match="Maximum iteration reached before convergence."
    ):
        estimator.fit(X_train, y_train, sample_domain=sample_domain)


def test_KMMReweight_kernel_error():
    with pytest.raises(ValueError, match="got 'hello'"):
        KMMReweightAdapter(kernel="hello")


# KMMReweight.adapt behavior should be the same when smooth weights is True or
# when X_source differs between fit and adapt.
def test_KMMReweight_new_X_adapt(da_dataset):
    X_train, y_train, sample_domain = da_dataset.pack_train(
        as_sources=["s"], as_targets=["t"]
    )
    estimator = KMMReweightAdapter(smooth_weights=True)
    estimator.fit(X_train, sample_domain=sample_domain)
    res1 = estimator.adapt(X_train, sample_domain=sample_domain)

    estimator = KMMReweightAdapter(smooth_weights=False)
    estimator.fit(X_train, sample_domain=sample_domain)
    res2 = estimator.adapt(X_train, sample_domain=sample_domain)
    res3 = estimator.adapt(X_train + 1e-8, sample_domain=sample_domain)

    assert np.allclose(res1["sample_weight"], res3["sample_weight"])
    assert not np.allclose(res1["sample_weight"], res2["sample_weight"])
