# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#
# License: BSD 3-Clause

import numpy as np
from numpy.testing import assert_array_equal

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from skada import SubspaceAlignmentAdapter, PerDomain, Shared, make_da_pipeline

import pytest


def test_pipeline(da_dataset):
    # single source, single target, target labels are masked
    X, y, sample_domain = da_dataset.pack_train(as_sources=['s'], as_targets=['t'])
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
    X_target, y_target, sample_domain = da_dataset.pack_test(as_targets=['t'])
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


def test_per_domain_selector():
    scaler = PerDomain(StandardScaler())
    X = np.array([[1., 0.], [0., 8.], [3., 0.], [0., 0.]])
    sample_domain = np.array([1, 2, 1, 2])
    scaler.fit(X, None, sample_domain=sample_domain)
    assert_array_equal(
        np.array([[-3., 1.]]),
        scaler.transform(np.array([[-1., 1.]]), sample_domain=[1])
    )
    assert_array_equal(
        np.array([[-1., -0.75]]),
        scaler.transform(np.array([[-1., 1.]]), sample_domain=[2])
    )


@pytest.mark.parametrize(
    'selector_name, selector_cls',
    [
        ('per_domain', PerDomain),
        ('shared', Shared),
        (PerDomain, PerDomain),
        (lambda x: PerDomain(x), PerDomain),
        pytest.param(
            'non_existing_one', None,
            marks=pytest.mark.xfail(reason='Fails non-existing selector')
        ),
        pytest.param(
            42, None,
            marks=pytest.mark.xfail(reason='Fails uninterpretable type')
        ),
        pytest.param(
            lambda x: 42, None,
            marks=pytest.mark.xfail(reason='Incorrect output type for the callable')
        )
    ]
)
def test_default_selector_parameter(selector_name, selector_cls):
    pipe = make_da_pipeline(
        SubspaceAlignmentAdapter(n_components=2),
        LogisticRegression(),
        default_selector=selector_name
    )
    _, estimator = pipe.steps[0]
    assert isinstance(estimator, selector_cls)


def test_default_selector_ignored_for_selector():
    pipe = make_da_pipeline(
        Shared(SubspaceAlignmentAdapter(n_components=2)),
        LogisticRegression(),
        default_selector='per_domain',
    )
    _, estimator = pipe.steps[0]
    assert isinstance(estimator, Shared)
    _, estimator = pipe.steps[1]
    assert isinstance(estimator, PerDomain)


def test_pipeline_step_parameters(da_dataset):
    pipe = make_da_pipeline(
        StandardScaler(),
        PCA(),
        SubspaceAlignmentAdapter(n_components=2),
        LogisticRegression(),
    )
    pipe.set_params(subspacealignmentadapter__n_components=5)
    with pytest.raises(ValueError):
        pipe.set_params(subspacealignmentadapter__reg=2.)
