# Author: Remi Flamary <remi.flamary@polytechnique.edu>
#
# License: BSD 3-Clause

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from skada import JDOTClassifier, JDOTRegressor, make_da_pipeline
from skada._ot import get_jdot_class_cost_matrix, get_tgt_loss_jdot_class
from skada.metrics import PredictionEntropyScorer
from skada.utils import source_target_split


def test_JDOTRegressor(da_reg_dataset):
    X, y, sample_domain = da_reg_dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=False
    )
    rng = np.random.default_rng(42)
    w = rng.uniform(size=(X.shape[0],))

    Xs, Xt, ys, yt = source_target_split(X, y, sample_domain=sample_domain)

    # standard case
    jdot = JDOTRegressor(base_estimator=Ridge(), alpha=0.1, verbose=True)
    jdot.fit(X, y, sample_domain=sample_domain)
    ypred = jdot.predict(Xt)
    assert ypred.shape[0] == Xt.shape[0]

    # JDOT with weights
    jdot = JDOTRegressor(base_estimator=Ridge(), verbose=True, n_iter_max=1)
    jdot.fit(X, y, sample_weight=w, sample_domain=sample_domain)
    score = jdot.score(X, y, sample_domain=sample_domain)
    assert score >= 0

    # JDOT with default base estimator
    jdot = JDOTRegressor()
    jdot.fit(X, y, sample_domain=sample_domain)

    with np.testing.assert_raises(ValueError):
        jdot = JDOTRegressor(StandardScaler())


def test_JDOTRegressor_pipeline(da_reg_dataset):
    X, y, sample_domain = da_reg_dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=False
    )
    Xs, Xt, ys, yt = source_target_split(X, y, sample_domain=sample_domain)

    jdot = make_da_pipeline(
        StandardScaler(), JDOTRegressor(Ridge(), alpha=0.1, verbose=True)
    )
    jdot.fit(X, y, sample_domain=sample_domain)

    ypred = jdot.predict(Xt)
    assert ypred.shape[0] == Xt.shape[0]

    ypred2 = jdot.predict(X, sample_domain=sample_domain)
    assert ypred2.shape[0] == X.shape[0]


def test_JDOTClassifier(da_multiclass_dataset, da_binary_dataset):
    rng = np.random.default_rng(43)
    for dataset in [da_multiclass_dataset, da_binary_dataset]:
        X, y, sample_domain = dataset.pack(
            as_sources=["s"], as_targets=["t"], mask_target_labels=False
        )
        w = rng.uniform(size=(X.shape[0],))
        Xs, Xt, ys, yt = source_target_split(X, y, sample_domain=sample_domain)

        # standard case (Logistic)
        jdot = JDOTClassifier(LogisticRegression(), alpha=0.1, verbose=True)
        jdot.fit(X, y, sample_domain=sample_domain)
        ypred = jdot.predict(Xt)
        assert ypred.shape[0] == Xt.shape[0]

        # JDOT with weights
        jdot = JDOTClassifier(
            base_estimator=SVC(), verbose=True, n_iter_max=1, metric="hinge"
        )
        jdot.fit(X, y, sample_weight=w, sample_domain=sample_domain)

        score = jdot.score(X, y, sample_domain=sample_domain)
        assert score >= 0

        # JDOT with default base estimator
        jdot = JDOTClassifier()
        jdot.fit(X, y, sample_domain=sample_domain)

        # No predict_proba method in base estimator
        with np.testing.assert_raises(AttributeError):
            jdot = JDOTClassifier(base_estimator=SVC(probability=False), metric="hinge")
            jdot.fit(X, y, sample_domain=sample_domain)
            _ = jdot.predict_proba(X)

        # with scorer needing predict_proba
        scorer = PredictionEntropyScorer()
        jdot = JDOTClassifier(base_estimator=SVC(probability=True))
        jdot.fit(X, y, sample_domain=sample_domain)
        scorer._score(jdot, X, y, sample_domain=sample_domain)

        # test raise error
        with np.testing.assert_raises(ValueError):
            jdot = JDOTClassifier(StandardScaler())

            # test raise error
        with np.testing.assert_raises(ValueError):
            jdot = JDOTClassifier(metric="bad_metric")
            jdot.fit(X, y, sample_domain=sample_domain)

        # No porba method in base estimator
        with np.testing.assert_raises(ValueError):
            jdot = JDOTClassifier(SVC())
            jdot.fit(X, y, sample_domain=sample_domain)

        # no decision function with hinge
        with np.testing.assert_raises(ValueError):
            jdot = JDOTClassifier(RandomForestClassifier(), metric="hinge")
            jdot.fit(X, y, sample_domain=sample_domain)

        # estimator without log_proba
        lp = LogisticRegression.predict_log_proba
        delattr(LogisticRegression, "predict_log_proba")
        jdot = JDOTClassifier(LogisticRegression())
        jdot.fit(X, y, sample_domain=sample_domain)
        setattr(LogisticRegression, "predict_log_proba", lp)


def test_jdot_class_cost_matrix(da_multiclass_dataset):
    X, y, sample_domain = da_multiclass_dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=False
    )

    Xs, Xt, ys, yt = source_target_split(X, y, sample_domain=sample_domain)

    Ys = OneHotEncoder().fit_transform(ys[:, None]).toarray()

    # test without predict log_proba
    lp = LogisticRegression.predict_log_proba
    delattr(LogisticRegression, "predict_log_proba")
    est = LogisticRegression()
    est.fit(Xs, ys)
    M = get_jdot_class_cost_matrix(Ys, Xt, est)
    setattr(LogisticRegression, "predict_log_proba", lp)
    assert M.shape[0] == Ys.shape[0]

    # raise because no probas
    with np.testing.assert_raises(ValueError):
        est = SVC()
        est.fit(Xs, ys)
        M = get_jdot_class_cost_matrix(Ys, Xt, est)

    # raise because no decision function
    with np.testing.assert_raises(ValueError):
        est = RandomForestClassifier()
        est.fit(Xs, ys)
        M = get_jdot_class_cost_matrix(Ys, Xt, est, metric="hinge")

    with np.testing.assert_raises(ValueError):
        est = LogisticRegression()
        est.fit(Xs, ys)
        M = get_jdot_class_cost_matrix(Ys, Xt, est, metric="bad_metric")


def test_jdot_class_tgt_loss(da_multiclass_dataset):
    X, y, sample_domain = da_multiclass_dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=False
    )

    Xs, Xt, ys, yt = source_target_split(X, y, sample_domain=sample_domain)

    Ys = OneHotEncoder().fit_transform(ys[:, None]).toarray()

    ws = np.ones(Xs.shape[0])

    # test without predict log_proba
    lp = LogisticRegression.predict_log_proba
    delattr(LogisticRegression, "predict_log_proba")
    est = LogisticRegression()
    est.fit(Xs, ys)
    loss = get_tgt_loss_jdot_class(Xs, Ys, ws, est)
    setattr(LogisticRegression, "predict_log_proba", lp)
    assert loss >= 0

    # raise because no probas
    with np.testing.assert_raises(ValueError):
        est = SVC()
        est.fit(Xs, ys)
        loss = get_tgt_loss_jdot_class(Xs, Ys, ws, est)

    # raise because no decision function
    with np.testing.assert_raises(ValueError):
        est = RandomForestClassifier()
        est.fit(Xs, ys)
        loss = get_tgt_loss_jdot_class(Xs, Ys, ws, est, metric="hinge")

    with np.testing.assert_raises(ValueError):
        est = LogisticRegression()
        est.fit(Xs, ys)
        loss = get_tgt_loss_jdot_class(Xs, Ys, ws, est, metric="bad_metric")
