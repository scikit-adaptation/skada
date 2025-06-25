# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Oleksii Kachaiev <kachayev@gmail.com>
#         Yanis Lalou <yanis.lalou@polytechnique.edu>
#         Antoine Collas <contact@antoinecollas.fr>
#
# License: BSD 3-Clause
import pytest

pytest.importorskip("torch")

import numpy as np
from sklearn.metrics import accuracy_score

from skada.datasets import make_shifted_datasets
from skada.deep import SourceOnly, TargetOnly
from skada.deep.modules import ToyModule2D


def test_sourceonly():
    n_samples = 20
    dataset = make_shifted_datasets(
        n_samples_source=n_samples,
        n_samples_target=n_samples,
        shift="conditional_shift",
        noise=0.1,
        mean=5,
        sigma=1,
        random_state=42,
        standardize=True,
        return_dataset=True,
    )

    method = SourceOnly(
        ToyModule2D(num_features=100),
        batch_size=10,
        max_epochs=100,
        train_split=None,
    )

    # Get full dataset without masking target
    X, y, sample_domain = dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=False
    )

    # Fit and predict
    method.fit(X.astype(np.float32), y, sample_domain)
    y_pred = method.predict(X.astype(np.float32), sample_domain, allow_source=True)

    assert y_pred.shape[0] == X.shape[0]

    history = method.history_

    assert history[0]["train_loss"] > history[-1]["train_loss"]

    # Check accuracy is better on source domain than on target domain
    X_source, y_source, sample_domain_source = dataset.pack(
        as_sources=["s"], as_targets=[], mask_target_labels=True
    )
    X_target, y_target, sample_domain_target = dataset.pack(
        as_sources=[], as_targets=["t"], mask_target_labels=False
    )
    y_pred_source = method.predict(
        X_source.astype(np.float32), sample_domain_source, allow_source=True
    )
    y_pred_target = method.predict(X_target.astype(np.float32), sample_domain_target)

    acc_source = accuracy_score(y_source, y_pred_source)
    acc_target = accuracy_score(y_target, y_pred_target)

    assert acc_source > 0.95
    assert acc_target < 0.55


def test_targetonly():
    n_samples = 20
    dataset = make_shifted_datasets(
        n_samples_source=n_samples,
        n_samples_target=n_samples,
        shift="conditional_shift",
        noise=0.1,
        mean=5,
        sigma=1,
        random_state=42,
        standardize=True,
        return_dataset=True,
    )

    method = TargetOnly(
        ToyModule2D(num_features=100),
        batch_size=10,
        max_epochs=100,
        train_split=None,
    )

    # Get full dataset without masking target
    X, y, sample_domain = dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=False
    )

    # Fit and predict
    method.fit(X.astype(np.float32), y, sample_domain)
    y_pred = method.predict(X.astype(np.float32), sample_domain, allow_source=True)

    assert y_pred.shape[0] == X.shape[0]

    history = method.history_

    assert history[0]["train_loss"] > history[-1]["train_loss"]

    # Check accuracy is better on target domain than on source domain
    X_source, y_source, sample_domain_source = dataset.pack(
        as_sources=["s"], as_targets=[], mask_target_labels=True
    )
    X_target, y_target, sample_domain_target = dataset.pack(
        as_sources=[], as_targets=["t"], mask_target_labels=False
    )
    y_pred_source = method.predict(
        X_source.astype(np.float32), sample_domain_source, allow_source=True
    )
    y_pred_target = method.predict(X_target.astype(np.float32), sample_domain_target)

    acc_source = accuracy_score(y_source, y_pred_source)
    acc_target = accuracy_score(y_target, y_pred_target)

    assert acc_source < 0.55
    assert acc_target > 0.95
