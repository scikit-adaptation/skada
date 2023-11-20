# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Oleksii Kachaiev <kachayev@gmail.com>
#
# License: BSD 3-Clause

import numpy as np
from numpy.testing import assert_array_equal

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from skada import make_da_pipeline, SubspaceAlignmentAdapter

import pytest


def test_pipeline(da_dataset):
    # single source, single target, target labels are masked
    X, y, sample_domain = da_dataset.pack_for_train(as_sources=['s'], as_targets=['t'])
    # by default, each estimator in the pipeline is wrapped into `Shared` selector
    pipe = make_da_pipeline(
        StandardScaler(),
        PCA(),
        SubspaceAlignmentAdapter(n_components=2),
        LogisticRegression(),
    )
    # 'sample_domain' is requested by the selector (`Shared` in this case)
    pipe.fit(X, y, sample_domain=sample_domain)
    # fails when passing source into predict
    with pytest.raises(ValueError):
        pipe.predict(X, sample_domain=sample_domain)
    with pytest.raises(ValueError):
        pipe.score(X, y, sample_domain=sample_domain)
    # target only, no label masking
    X_target, y_target, sample_domain = da_dataset.pack_for_test(as_targets=['t'])
    y_pred = pipe.predict(X_target, sample_domain=sample_domain)
    assert np.mean(y_pred == y_target) > 0.9
    # automatically derives as a single target domain when sample_domain is `None`
    y_pred_derived = pipe.predict(X_target)
    assert_array_equal(y_pred, y_pred_derived, 'automatically derives as target')
    # proxy to the default scoring of the final estimator
    # (`LogisticRegression` in this case)
    score = pipe.score(X_target, y_target, sample_domain=sample_domain)
    assert score > 0.9
    score_derived = pipe.score(X_target, y_target)
    assert score == score_derived, 'automatically derives as target'
