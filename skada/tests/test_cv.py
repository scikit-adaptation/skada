# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#
# License: BSD 3-Clause

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GroupShuffleSplit,
    GroupKFold,
    LeaveOneGroupOut,
    cross_validate,
)
import imp

from skada import SubspaceAlignmentAdapter, make_da_pipeline
from skada.metrics import PredictionEntropyScorer
from skada.model_selection import LeaveOneDomainOut, SourceTargetShuffleSplit

import pytest


# @pytest.mark.parametrize(
#     'cv, n_splits',
#     [
#         (GroupShuffleSplit(n_splits=2, test_size=0.3, random_state=0), 2),
#         (GroupKFold(n_splits=2), 2),
#         (LeaveOneGroupOut(), 4),
#     ]
# )
@pytest.mark.parametrize(
    'cv, n_splits',
    [
        (GroupKFold(n_splits=2), 2),
        (LeaveOneGroupOut(), 4),
    ]
)
def test_group_based_cv(da_dataset, cv, n_splits):
    assert hasattr(cv, 'set_split_request'), 'splitter has to support routing'
    cv.set_split_request(groups='sample_domain')
    # single source, single target, target labels are masked
    X, y, sample_domain = da_dataset.pack_train(
        as_sources=['s', 's2'],
        as_targets=['t', 't2']
    )
    # by default, each estimator in the pipeline is wrapped into `Shared` selector
    pipe = make_da_pipeline(
        SubspaceAlignmentAdapter(n_components=2),
        LogisticRegression(),
    )
    #pipe.fit(X, y, sample_domain=sample_domain)
    scores = cross_validate(
        pipe,
        X,
        y,
        cv=cv,
        params={'sample_domain': sample_domain},
        scoring=PredictionEntropyScorer(),
        error_score="raise",
    )['test_score']

    # gss = GroupShuffleSplit(n_splits=2, train_size=.3, random_state=0)
    # splits = list(gss.split(X, y, sample_domain))
    # indices = list(cv._iter_indices(X, y, sample_domain))

    # scorer = PredictionEntropyScorer()
    #import pdb; pdb.set_trace()

    # print("X split len: ", len(X[splits[0][0]]))
    # print("y split len: ", len(y[splits[0][0]]))
    # print("sample_domain split len: ", len(sample_domain[splits[0][0]]))
    # pipe.fit(X[splits[0][0]], y[splits[0][0]], sample_domain=sample_domain[splits[0][0]])
    # scorer._score(pipe, X[splits[0][1]], y[splits[0][1]], sample_domain[splits[0][1]])

    assert scores.shape[0] == n_splits, "evaluate all splits"
    # xxx(okachaiev): make sure we understand why/when validation fails
    # (some results are certainly None in here)
    assert np.any(~np.isnan(scores)), "at least some scores are computed"


    # # Test the split method
    # splits = list(cv.split(X, y, sample_domain))
    # assert len(splits) == 2
    # assert len(splits[0]) == len(splits[1]) == 2
    # assert len(splits[0][0]) == len(X)/len(splits)

    # # Test the _iter_indices method
    # indices = list(cv._iter_indices(X, y, sample_domain))
    # assert len(indices) == 2
    # assert len(indices[0]) == len(indices[1]) == 2
    # assert len(indices[0][0]) == len(X)/len(indices)


# def test_domain_aware_split(da_dataset):
#     X, y, sample_domain = da_dataset.pack_train(
#         as_sources=['s', 's2'],
#         as_targets=['t', 't2']
#     )
#     pipe = make_da_pipeline(
#         SubspaceAlignmentAdapter(n_components=2),
#         LogisticRegression(),
#     )
#     pipe.fit(X, y, sample_domain=sample_domain)
#     n_splits = 4
#     cv = SourceTargetShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=0)
#     scores = cross_validate(
#         pipe,
#         X,
#         y,
#         cv=cv,
#         params={'sample_domain': sample_domain},
#         scoring=PredictionEntropyScorer(),
#     )['test_score']

#     gss = SourceTargetShuffleSplit(n_splits=2, train_size=.3, random_state=0)
#     splits = list(gss.split(X, y, sample_domain))
#     indices = list(cv._iter_indices(X, y, sample_domain))
#     assert scores.shape[0] == n_splits, "evaluate all splits"
#     assert np.all(~np.isnan(scores)), "at least some scores are computed"


@pytest.mark.parametrize(
    'max_n_splits, n_splits',
    [(2, 2), (10, 4)]
)
def test_leave_one_domain_out(da_dataset, max_n_splits, n_splits):
    X, y, sample_domain = da_dataset.pack_lodo()
    pipe = make_da_pipeline(
        SubspaceAlignmentAdapter(n_components=2),
        LogisticRegression(),
    )
    pipe.fit(X, y, sample_domain=sample_domain)
    cv = LeaveOneDomainOut(max_n_splits=max_n_splits, test_size=0.3, random_state=0)
    scores = cross_validate(
        pipe,
        X,
        y,
        cv=cv,
        params={'sample_domain': sample_domain},
        scoring=PredictionEntropyScorer(),
    )['test_score']
    assert scores.shape[0] == n_splits, "evaluate all splits"
    assert np.all(~np.isnan(scores)), "at least some scores are computed"
