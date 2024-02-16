# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#
# License: BSD 3-Clause

import numpy as np
from sklearn.linear_model import LogisticRegression

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
    make_da_pipeline,
)

import pytest


# xxx(okachaiev): the problem with the pipeline being setup this way,
#                 the estimator does not accept sample_weights (as it
#                 doesn't request it from the routing)
@pytest.mark.parametrize(
    "estimator",
    [
        make_da_pipeline(ReweightDensityAdapter(), LogisticRegression()),
        ReweightDensity(),
        make_da_pipeline(GaussianReweightDensityAdapter(), LogisticRegression()),
        GaussianReweightDensity(),
        make_da_pipeline(DiscriminatorReweightDensityAdapter(), LogisticRegression()),
        DiscriminatorReweightDensity(),
        make_da_pipeline(
            KLIEPAdapter(gamma=[0.1, 1], random_state=42),
            LogisticRegression()
        ),
        KLIEP(gamma=[0.1, 1], random_state=42),
        KLIEP(gamma=0.2),
        make_da_pipeline(MMDTarSReweightAdapter(1.0), LogisticRegression()),
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
