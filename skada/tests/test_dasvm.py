# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#
# License: BSD 3-Clause

from skada.datasets import make_shifted_datasets
from skada._dasvm import BaseDasvmAdapter
from skada.utils import check_X_y_domain, source_target_split


def test_mapping_estimator():
    X, y, sample_domain = make_shifted_datasets(
        n_samples_source=12,
        n_samples_target=10,
        shift="covariate_shift",
        noise=None,
        label="binary",
    )
    X, y, sample_domain = check_X_y_domain(X, y, sample_domain)
    Xs, Xt, ys, yt = source_target_split(
        X, y, sample_domain=sample_domain
    )

    clf_dsvm = BaseDasvmAdapter(k=5).fit(X, y, sample_domain)

    assert clf_dsvm.get_estimator().n_features_in_ == 2, (
        "Obtained estimator take the wrong number of features"
        )
    Xt_, yt_ = clf_dsvm.adapt(X, y, sample_domain)
    assert Xt_.shape == Xt.shape, (
        "Wrong shape of the target features when using adapt method"
        )
    assert yt_.shape == yt.shape, (
        "Wrong shape of the target y-values (labels) when using adapt method"
        )

    # Still not implemented
    clf_dsvm.predict(Xt)
