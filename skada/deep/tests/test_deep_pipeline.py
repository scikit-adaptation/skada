# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#
# License: BSD 3-Clause
import pytest

pytest.importorskip("torch")

import numpy as np
import pytest

from sklearn.preprocessing import StandardScaler

from skada.deep import DeepCoral
from skada.deep.modules import ToyModule2D
from skada import make_da_pipeline


def test_deepcoral(da_dataset):
    module = ToyModule2D()
    module.eval()

    X, y, sample_domain = da_dataset.pack_train(as_sources=["s"], as_targets=["t"])
    model = DeepCoral(
        ToyModule2D(),
        reg=1,
        layer_name="dropout",
        batch_size=10,
        max_epochs=10,
        train_split=None,
    )
    pipe = make_da_pipeline(
        # StandardScaler(),
        DeepCoral(
            ToyModule2D(),
            reg=1,
            layer_name="dropout",
            batch_size=10,
            max_epochs=10,
            train_split=None,
        ),
    )
    # model.fit(X.astype(np.float32), y, sample_domain=sample_domain)
    pipe.fit(X.astype(np.float32), y, sample_domain=sample_domain)

    # X_test, y_test, sample_domain_test = dataset.pack_test(as_targets=["t"])

    # y_pred = method.predict(X_test.astype(np.float32), sample_domain_test)

    # assert y_pred.shape[0] == X_test.shape[0]

    # history = method.history_

    # assert history[0]["train_loss"] > history[-1]["train_loss"]

