# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Oleksii Kachaiev <kachayev@gmail.com>
#
# License: BSD 3-Clause
import pytest

torch = pytest.importorskip("torch")

import numpy as np
from torch.nn import BCELoss

from skada.datasets import make_shifted_datasets
from skada.deep import SPA
from skada.deep.modules import DomainClassifier, ToyModule2D


@pytest.mark.parametrize(
    "domain_classifier, domain_criterion, num_features",
    [
        (DomainClassifier(num_features=10), BCELoss(), 10),
        (DomainClassifier(num_features=10), None, 10),
        (None, None, 10),
        (None, BCELoss(), 10),
    ],
)
def test_spa(domain_classifier, domain_criterion, num_features):
    module = ToyModule2D()
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
    X, y, sample_domain = dataset.pack_train(as_sources=["s"], as_targets=["t"])
    n_samples_t = np.sum(sample_domain < 0)
    method = SPA(
        ToyModule2D(),
        domain_classifier=domain_classifier,
        num_features=num_features,
        num_classes=2,
        num_samples_t=n_samples_t,
        domain_criterion=domain_criterion,
        layer_name="dropout",
        batch_size=10,
        max_epochs=10,
        train_split=None,
    )

    method.fit(X.astype(np.float32), y, sample_domain)

    X_test, y_test, sample_domain_test = dataset.pack_test(as_targets=["t"])

    y_pred = method.predict(X_test.astype(np.float32), sample_domain_test)

    assert y_pred.shape[0] == X_test.shape[0]

    history = method.history_

    assert history[0]["train_loss"] > history[-1]["train_loss"]