# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Oleksii Kachaiev <kachayev@gmail.com>
#
# License: BSD 3-Clause
import pytest

pytest.importorskip("torch")

import numpy as np

from skada.datasets import make_shifted_datasets
from skada.deep import DeepJDOT
from skada.deep.modules import ToyModule2D


def test_deepjdot():
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

    method = DeepJDOT(
        ToyModule2D(),
        reg_dist=1,
        reg_cl=1,
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
