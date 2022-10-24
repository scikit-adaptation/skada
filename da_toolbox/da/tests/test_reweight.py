import numpy as np
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression

from da_toolbox.da.reweight import ReweightDensity


def test_reweight_density():
    centers = np.array([
        [0, 0],
        [1, 1],
    ])
    n_classes, n_features = centers.shape
    rng = np.random.RandomState(42)
    X, y = make_blobs(
        n_samples=500,
        centers=centers,
        n_features=n_features,
        random_state=rng,
        cluster_std=0.05,
    )

    X_target, y_target = make_blobs(
        n_samples=200,
        centers=centers,
        n_features=n_features,
        random_state=rng,
        cluster_std=0.05,
    )
    X_target[:, 0] += 0.13
    X_target[:, 1] += 0.13

    clf = ReweightDensity(base_estimator=LogisticRegression())
    clf.fit(X, y, X_target)
    y_pred = clf.predict(X_target)
    assert np.mean(y_pred == y_target) > 0.9
    score = clf.score(X_target, y_target)
    assert score > 0.9
