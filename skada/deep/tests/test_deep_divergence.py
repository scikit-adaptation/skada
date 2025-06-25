# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Oleksii Kachaiev <kachayev@gmail.com>
#         Yanis Lalou <yanis.lalou@polytechnique.edu>
#
# License: BSD 3-Clause
import pytest
from skorch.callbacks import EpochScoring

pytest.importorskip("torch")

import numpy as np
import torch

from skada.datasets import make_shifted_datasets
from skada.deep import CAN, DAN, DeepCoral
from skada.deep.losses import cdd_loss, dan_loss
from skada.deep.modules import ToyModule2D
from skada.deep.utils import SphericalKMeans


@pytest.mark.parametrize(
    "assume_centered",
    [
        True,
        False,
    ],
)
def test_deepcoral(assume_centered):
    module = ToyModule2D()
    module.eval()

    n_samples = 20
    dataset = make_shifted_datasets(
        n_samples_source=n_samples,
        n_samples_target=n_samples,
        shift="conditional_shift",
        noise=0.1,
        random_state=42,
        return_dataset=True,
    )

    method = DeepCoral(
        ToyModule2D(),
        reg=1,
        layer_name="dropout",
        batch_size=15,
        max_epochs=10,
        train_split=None,
        assume_centered=assume_centered,
    )

    X, y, sample_domain = dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )
    method.fit(X.astype(np.float32), y, sample_domain)

    X_test, y_test, sample_domain_test = dataset.pack(
        as_sources=[], as_targets=["t"], mask_target_labels=False
    )

    y_pred = method.predict(X_test.astype(np.float32), sample_domain_test)

    assert y_pred.shape[0] == X_test.shape[0]

    history = method.history_

    assert history[0]["train_loss"] > history[-1]["train_loss"]


@pytest.mark.parametrize(
    "sigmas",
    [
        [0.1, 0.2],
        None,
    ],
)
def test_dan(sigmas):
    module = ToyModule2D()
    module.eval()

    n_samples = 20
    dataset = make_shifted_datasets(
        n_samples_source=n_samples,
        n_samples_target=n_samples,
        shift="conditional_shift",
        noise=0.1,
        random_state=42,
        return_dataset=True,
    )

    method = DAN(
        ToyModule2D(),
        reg=1,
        sigmas=sigmas,
        layer_name="dropout",
        batch_size=10,
        max_epochs=10,
        train_split=None,
    )

    X, y, sample_domain = dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )
    method.fit(X.astype(np.float32), y, sample_domain)

    X_test, y_test, sample_domain_test = dataset.pack(
        as_sources=[], as_targets=["t"], mask_target_labels=False
    )

    y_pred = method.predict(X_test.astype(np.float32), sample_domain_test)

    assert y_pred.shape[0] == X_test.shape[0]

    history = method.history_

    assert history[0]["train_loss"] > history[-1]["train_loss"]


@pytest.mark.parametrize(
    "sigmas, distance_threshold, class_threshold",
    [
        ([0.1, 0.2], 0.5, 3),
        (None, 0.5, 3),
        ([0.1, 0.2], 0.2, 5),
    ],
)
def test_can(sigmas, distance_threshold, class_threshold):
    module = ToyModule2D()
    module.eval()

    n_samples = 10
    dataset = make_shifted_datasets(
        n_samples_source=n_samples,
        n_samples_target=n_samples,
        shift="conditional_shift",
        noise=0.1,
        random_state=42,
        return_dataset=True,
    )

    method = CAN(
        ToyModule2D(),
        reg=0.01,
        sigmas=sigmas,
        distance_threshold=distance_threshold,
        class_threshold=class_threshold,
        layer_name="dropout",
        batch_size=10,
        max_epochs=10,
        train_split=None,
    )

    X, y, sample_domain = dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )
    method.fit(X.astype(np.float32), y, sample_domain)

    X_test, y_test, sample_domain_test = dataset.pack(
        as_sources=[], as_targets=["t"], mask_target_labels=False
    )

    y_pred = method.predict(X_test.astype(np.float32), sample_domain_test)

    assert y_pred.shape[0] == X_test.shape[0]

    history = method.history_
    assert history[0]["train_loss"] > history[-1]["train_loss"]


def test_can_with_custom_callbacks():
    module = ToyModule2D()
    module.eval()

    n_samples = 10
    dataset = make_shifted_datasets(
        n_samples_source=n_samples,
        n_samples_target=n_samples,
        shift="conditional_shift",
        noise=0.1,
        random_state=42,
        return_dataset=True,
    )

    # Create a custom callback
    custom_callback = EpochScoring(scoring="accuracy", lower_is_better=False)

    method = CAN(
        ToyModule2D(),
        reg=0.01,
        layer_name="dropout",
        batch_size=10,
        max_epochs=10,
        train_split=None,
        callbacks=[custom_callback],  # Pass the custom callback
    )

    X, y, sample_domain = dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )
    method.fit(X.astype(np.float32), y, sample_domain)

    X_test, y_test, sample_domain_test = dataset.pack(
        as_sources=[], as_targets=["t"], mask_target_labels=False
    )

    y_pred = method.predict(X_test.astype(np.float32), sample_domain_test)

    assert y_pred.shape[0] == X_test.shape[0]

    history = method.history_
    assert history[0]["train_loss"] > history[-1]["train_loss"]

    # Check if both custom callback and ComputeSourceCentroids are present
    callback_classes = [cb.__class__.__name__ for cb in method.callbacks]
    assert "EpochScoring" in callback_classes
    assert "ComputeSourceCentroids" in callback_classes


def test_cdd_loss_edge_cases():
    # Test when median pairwise distance is 0
    features_s = torch.ones((4, 2))  # All features are identical
    features_t = torch.randn((4, 2))
    y_s = torch.tensor([0, 0, 1, 1])  # Two classes

    # Fit a SphericalKMeans
    target_kmeans = SphericalKMeans(
        n_clusters=len(torch.unique(y_s)),
        random_state=0,
    )
    target_kmeans.fit(features_t)

    # This should not raise any errors due to the eps we added
    loss = cdd_loss(y_s, features_s, features_t, target_kmeans)
    assert not np.isnan(loss)


def test_dan_loss_edge_cases():
    # Create identical source features to get median distance = 0
    features_s = torch.tensor([[1.0, 2.0], [1.0, 2.0]], dtype=torch.float32)
    features_t = torch.tensor([[3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)

    # Verify median distance is 0
    assert torch.median(torch.cdist(features_s, features_s)) == 0

    # Test that dan_loss still works
    loss = dan_loss(features_s, features_t)

    # Loss should be finite and non-negative
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss >= 0
