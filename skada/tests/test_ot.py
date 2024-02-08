# Author: Remi Flamary <remi.flamary@polytechnique.edu>
#
# License: BSD 3-Clause


from skada import JDOTRegressor
from sklearn.linear_model import Ridge
from skada.utils import source_target_split


def test_JDOTRegressor(da_reg_dataset):

    X, y, sample_domain = da_reg_dataset

    Xs, Xt, ys, yt = source_target_split(X, y, sample_domain=sample_domain)

    jdot = JDOTRegressor(base_estimator=Ridge(), verbose=True)

    jdot.fit(X, y, sample_domain)

    ypred = jdot.predict(Xt)

    assert ypred.shape[0] == Xt.shape[0]

    jdot = JDOTRegressor(base_estimator=Ridge(), verbose=True, n_iter_max=1)
    jdot.fit(X, y, sample_domain)
