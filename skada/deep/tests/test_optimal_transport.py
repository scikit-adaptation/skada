# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#
# License: BSD 3-Clause

import numpy as np

from skada.deep import DeepJDOT
from skada.deep.modules import ToyModule
from skada.datasets import make_shifted_datasets


def test_deepjdot():
    module = ToyModule()
    module.eval()

    n_samples = 20
    dataset = make_shifted_datasets(
        n_samples_source=n_samples,
        n_samples_target=n_samples,
        shift="concept_drift",
        noise=0.1,
        random_state=42,
        return_dataset=True,
    )

    method = DeepJDOT(
        ToyModule(),
        reg_d=1,
        reg_cl=1,
        layer_name="dropout",
        batch_size=10,
        max_epochs=10,
        train_split=None
    )

    X, y, sample_domain = dataset.pack_train(as_sources=["s"], as_targets=["t"])
    method.fit(X.astype(np.float32), y, sample_domain)

    X_test, y_test, sample_domain_test = dataset.pack_test(as_targets=["t"])

    y_pred = method.predict(X_test.astype(np.float32), sample_domain_test)

    assert y_pred.shape[0] == X_test.shape[0]

    history = method.history_

    assert history[0]["train_loss"] > history[-1]["train_loss"]
