# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Oleksii Kachaiev <kachayev@gmail.com>
#         Yanis Lalou <yanis.lalou@polytechnique.edu>
#
# License: BSD 3-Clause
import pytest

pytest.importorskip("torch")

import numpy as np
import torch

from skada.datasets import make_shifted_datasets
from skada.deep import MCC
from skada.deep.losses import mcc_loss
from skada.deep.modules import ToyModule2D


@pytest.mark.parametrize(
    "T",
    [1, 0.5],
)
def test_mcc(T):
    module = ToyModule2D(n_classes=5)
    module.eval()

    n_samples = 50
    dataset = make_shifted_datasets(
        n_samples_source=n_samples,
        n_samples_target=n_samples,
        shift="conditional_shift",
        noise=0.1,
        random_state=42,
        return_dataset=True,
        label="multiclass",
    )

    method = MCC(
        module,
        reg=1,
        layer_name="dropout",
        batch_size=32,
        max_epochs=5,
        train_split=None,
        T=T,
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


def test_mcc_with_zeros():
    """Test that mcc_loss handles zero probabilities correctly."""
    # Create logits with extreme values that will result in zeros
    # after softmax operation due to numerical underflow
    logits = torch.tensor(
        [
            [100.0, -100.0, -100.0],
            [-100.0, 100.0, -100.0],
            [-100.0, -100.0, 100.0],
        ]
    )

    # Verify that we actually get zeros in y_scaled
    y_scaled = torch.nn.functional.softmax(logits, dim=1)
    assert torch.sum(y_scaled == 0.0) > 0, "Test setup failed: no zeros in y_scaled"

    # This should not raise any errors due to the epsilon in log
    loss = mcc_loss(logits, T=1.0)

    assert torch.isfinite(loss)  # Check that the loss is not NaN or infinite
    assert loss >= 0  # MCC loss should be non-negative
