# Author: Remi Flamary <remi.flamary@polytechnique.edu>
#
# License: BSD 3-Clause

import numpy as np
from skada import JDOTRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from skada.utils import source_target_split
from skada import make_da_pipeline


def test_JDOTRegressor(da_reg_dataset):

    X, y, sample_domain = da_reg_dataset
    w = np.random.rand(X.shape[0])

    Xs, Xt, ys, yt = source_target_split(X, y, sample_domain=sample_domain)

    jdot = JDOTRegressor(base_estimator=Ridge(), alpha=.1, verbose=True)

    jdot.fit(X, y, sample_domain=sample_domain)

    ypred = jdot.predict(Xt)

    assert ypred.shape[0] == Xt.shape[0]

    jdot = JDOTRegressor(base_estimator=Ridge(), verbose=True, n_iter_max=1)
    jdot.fit(X, y, sample_weight=w, sample_domain=sample_domain)

    score = jdot.score(X, y, sample_domain=sample_domain)

    assert score >= 0


def test_JDOTRegressor_pipeline(da_reg_dataset):

    X, y, sample_domain = da_reg_dataset

    Xs, Xt, ys, yt = source_target_split(X, y, sample_domain=sample_domain)

    jdot = make_da_pipeline(
        StandardScaler(), JDOTRegressor(
            Ridge(), alpha=.1, verbose=True))

    jdot.fit(X, y, sample_domain=sample_domain)

    ypred = jdot.predict(Xt)

    assert ypred.shape[0] == Xt.shape[0]

    ypred2 = jdot.predict(X, sample_domain=sample_domain)

    assert ypred2.shape[0] == X.shape[0]
