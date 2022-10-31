import numpy as np
from sklearn.linear_model import LogisticRegression

from skada.datasets import make_shifted_blobs
from skada.da.reweight import (
    ReweightDensity, GaussianReweightDensity, ClassifierReweightDensity
)


def test_reweight_density():
    centers = np.array([
        [0, 0],
        [1, 1],
    ])
    _, n_features = centers.shape

    X, y, X_target, y_target = make_shifted_blobs(
        n_samples=500,
        centers=centers,
        n_features=n_features,
        shift=0.13,
        random_state=42,
        cluster_std=0.05,
    )

    clf = ReweightDensity(base_estimator=LogisticRegression())
    clf.fit(X, y, X_target)
    y_pred = clf.predict(X_target)
    assert np.mean(y_pred == y_target) > 0.9
    score = clf.score(X_target, y_target)
    assert score > 0.9


def test_gaussian_reweight_density():
    centers = np.array([
        [0, 0],
        [1, 1],
    ])
    _, n_features = centers.shape

    X, y, X_target, y_target = make_shifted_blobs(
        n_samples=500,
        centers=centers,
        n_features=n_features,
        shift=0.13,
        random_state=42,
        cluster_std=0.05,
    )

    clf = GaussianReweightDensity(base_estimator=LogisticRegression())
    clf.fit(X, y, X_target)
    y_pred = clf.predict(X_target)
    assert np.mean(y_pred == y_target) > 0.9
    score = clf.score(X_target, y_target)
    assert score > 0.9


def test_classifier_reweight_density():
    centers = np.array([
        [0, 0],
        [1, 1],
    ])
    _, n_features = centers.shape

    X, y, X_target, y_target = make_shifted_blobs(
        n_samples=500,
        centers=centers,
        n_features=n_features,
        shift=0.13,
        random_state=42,
        cluster_std=0.05,
    )

    clf = ClassifierReweightDensity(base_estimator=LogisticRegression())
    clf.fit(X, y, X_target)
    y_pred = clf.predict(X_target)
    assert np.mean(y_pred == y_target) > 0.9
    score = clf.score(X_target, y_target)
    assert score > 0.9
