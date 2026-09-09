# Author: Christopher Engoang <christopherengoangperrat@gmail.com>
#
# License: BSD 3-Clause
import pytest

torch = pytest.importorskip("torch")

import numpy as np
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from skada import make_da_pipeline
from skada.deep import M3SDA, M3SDALoss


def _make_gaussian_domains(n_samples=16, random_state=42):
    generator = np.random.default_rng(random_state)

    def make_domain(class_zero_mean, class_one_mean):
        class_zero = generator.normal(class_zero_mean, 0.25, (n_samples, 2))
        class_one = generator.normal(class_one_mean, 0.25, (n_samples, 2))
        return (
            np.vstack([class_zero, class_one]).astype(np.float32),
            np.array([2] * n_samples + [4] * n_samples),
        )

    X_source_1, y_source_1 = make_domain([-2, 0], [-2, 2])
    X_source_2, y_source_2 = make_domain([0, -2], [2, -2])
    X_target, _ = make_domain([1, 1], [3, 3])

    X = np.vstack([X_source_1, X_source_2, X_target])
    y = np.concatenate(
        [
            y_source_1,
            y_source_2,
            np.full(X_target.shape[0], -1),
        ]
    )
    sample_domain = np.concatenate(
        [
            np.full(X_source_1.shape[0], 2),
            np.full(X_source_2.shape[0], 7),
            np.full(X_target.shape[0], -3),
        ]
    )
    return X, y, sample_domain, X_target


def test_m3sda_loss_is_zero_for_identical_domains():
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    loss = M3SDALoss(moment_order=2)([features, features], features)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_m3sda_loss_contains_source_source_term():
    source_1 = torch.zeros((2, 1))
    source_2 = torch.ones((2, 1))
    target = torch.zeros((2, 1))

    loss = M3SDALoss(moment_order=1)([source_1, source_2], target)

    # source-target: (0 + 1) / 2; source-source: 1 * abs(0 - 1)
    assert torch.isclose(loss, torch.tensor(1.5))


def test_m3sda_loss_aligns_second_moment():
    source_1 = torch.tensor([[-1.0], [1.0]])
    source_2 = torch.tensor([[-2.0], [2.0]])
    target = torch.tensor([[-1.0], [1.0]])

    first_order = M3SDALoss(moment_order=1)([source_1, source_2], target)
    second_order = M3SDALoss(moment_order=2)([source_1, source_2], target)

    assert torch.isclose(first_order, torch.tensor(0.0))
    assert second_order > first_order


def test_m3sda_fit_predict_and_clone():
    X, y, sample_domain, X_target = _make_gaussian_domains()
    estimator = M3SDA(
        hidden_dim=8,
        max_epochs=2,
        batch_size=8,
        random_state=0,
        device="cpu",
    )

    clone(estimator)
    estimator.fit(X, y, sample_domain=sample_domain)
    probabilities = estimator.predict_proba(
        X_target, sample_domain=np.full(X_target.shape[0], -3)
    )
    predictions = estimator.predict(X_target)
    features = estimator.transform(X_target)

    assert estimator.classes_.tolist() == [2, 4]
    assert estimator.source_domains_.tolist() == [2, 7]
    assert len(estimator.module_.classifiers) == 2
    assert probabilities.shape == (X_target.shape[0], 2)
    assert np.allclose(probabilities.sum(axis=1), 1)
    assert predictions.shape == (X_target.shape[0],)
    assert features.shape == (X_target.shape[0], 8)
    assert np.isfinite(estimator.loss_history_).all()


def test_m3sda_ignores_target_labels():
    X, y, sample_domain, X_target = _make_gaussian_domains()
    exposed_target_y = y.copy()
    exposed_target_y[sample_domain < 0] = 99
    parameters = dict(
        hidden_dim=8,
        max_epochs=1,
        batch_size=8,
        random_state=0,
        device="cpu",
    )

    masked_model = M3SDA(**parameters).fit(X, y, sample_domain=sample_domain)
    exposed_model = M3SDA(**parameters).fit(
        X, exposed_target_y, sample_domain=sample_domain
    )

    assert np.allclose(
        masked_model.predict_proba(X_target),
        exposed_model.predict_proba(X_target),
    )


def test_m3sda_in_da_pipeline():
    X, y, sample_domain, X_target = _make_gaussian_domains()
    model = M3SDA(
        hidden_dim=8,
        max_epochs=1,
        batch_size=8,
        random_state=0,
        device="cpu",
    )
    pipeline = make_da_pipeline(StandardScaler(), model)

    pipeline.fit(X, y, sample_domain=sample_domain)
    predictions = pipeline.predict(
        X_target, sample_domain=np.full(X_target.shape[0], -3)
    )

    assert predictions.shape == (X_target.shape[0],)


@pytest.mark.parametrize(
    "sample_domain, message",
    [
        (np.array([1, 1, -1, -1]), "at least two source domains"),
        (np.array([1, 2, 3, 4]), "exactly one target domain"),
    ],
)
def test_m3sda_rejects_invalid_domain_layout(sample_domain, message):
    X = np.arange(8, dtype=np.float32).reshape(4, 2)
    y = np.array([0, 1, -1, -1])
    estimator = M3SDA(max_epochs=1)

    with pytest.raises(ValueError, match=message):
        estimator.fit(X, y, sample_domain=sample_domain)
