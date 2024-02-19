# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#
# License: BSD 3-Clause

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

from skada import source_target_split
from skada import (
    ReweightDensityAdapter,
    ReweightDensity,
    GaussianReweightDensityAdapter,
    GaussianReweightDensity,
    DiscriminatorReweightDensityAdapter,
    DiscriminatorReweightDensity,
    KLIEPAdapter,
    KLIEP,
    MMDTarSReweightAdapter,
    MMDTarSReweight,
    make_da_pipeline,
)

import pytest


@pytest.mark.parametrize(
    "estimator",
    [
        make_da_pipeline(
            ReweightDensityAdapter(),
            LogisticRegression().set_fit_request(sample_weight=True)
        ),
        ReweightDensity(),
        make_da_pipeline(
            GaussianReweightDensityAdapter(),
            LogisticRegression().set_fit_request(sample_weight=True)
        ),
        GaussianReweightDensity(),
        make_da_pipeline(
            DiscriminatorReweightDensityAdapter(),
            LogisticRegression().set_fit_request(sample_weight=True)
        ),
        DiscriminatorReweightDensity(),
        make_da_pipeline(
            KLIEPAdapter(gamma=[0.1, 1], random_state=42),
            LogisticRegression().set_fit_request(sample_weight=True)
        ),
        KLIEP(gamma=[0.1, 1], random_state=42),
        KLIEP(gamma=0.2),
        make_da_pipeline(
            MMDTarSReweightAdapter(gamma=1.0),
            LogisticRegression().set_fit_request(sample_weight=True)
        ),
        MMDTarSReweight(gamma=1.0)
    ],
)
def test_reweight_estimator(estimator, da_dataset):
    X_train, y_train, sample_domain = da_dataset.pack_train(
        as_sources=['s'],
        as_targets=['t']
    )
    estimator.fit(X_train, y_train, sample_domain=sample_domain)
    X_test, y_test, sample_domain = da_dataset.pack_test(as_targets=['t'])
    y_pred = estimator.predict(X_test, sample_domain=sample_domain)
    assert np.mean(y_pred == y_test) > 0.9
    score = estimator.score(X_test, y_test, sample_domain=sample_domain)
    assert score > 0.9


@pytest.mark.parametrize(
    "estimator",
    [
        make_da_pipeline(
            MMDTarSReweightAdapter(gamma=1.0),
            Ridge().set_fit_request(sample_weight=True)
        ),
        MMDTarSReweight(
            gamma=1.0,
            base_estimator=Ridge().set_fit_request(sample_weight=True)
        )
    ],
)
def test_reg_reweight_estimator(estimator, da_reg_dataset):
    X, y, sample_domain = da_reg_dataset
    Xs, Xt, ys, yt = source_target_split(X, y, sample_domain=sample_domain)
    estimator.fit(X, y, sample_domain=sample_domain)
    score = estimator.score(Xt, yt)
    assert score >= 0


def test_reweight_warning(da_dataset):
    X_train, y_train, sample_domain = da_dataset.pack_train(
        as_sources=['s'],
        as_targets=['t']
    )
    estimator = KLIEPAdapter(gamma=0.1, max_iter=0)
    estimator.fit(X_train, y_train, sample_domain=sample_domain)

    with pytest.warns(UserWarning,
                      match="Maximum iteration reached before convergence."):
        estimator.fit(X_train, y_train, sample_domain=sample_domain)
