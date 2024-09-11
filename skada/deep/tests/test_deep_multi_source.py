# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#
# License: BSD 3-Clause
import pytest

pytest.importorskip("torch")
import numpy as np
import torch

from skada.datasets import make_dataset_from_moons_distribution
from skada.deep import MFSAN, MultiSourceModule


@pytest.mark.parametrize(
    "sigmas",
    [
        [0.1, 0.2],
        None,
    ],
)
def test_mfsan(sigmas):
    class CommonFeatureModule(torch.nn.Module):
        def __init__(self, num_features=10, nonlin=torch.nn.ReLU()):
            super().__init__()
            self.dense0 = torch.nn.Linear(2, num_features)
            self.nonlin = nonlin
            self.dropout = torch.nn.Dropout(0.5)

        def forward(self, X):
            X = self.nonlin(self.dense0(X))
            X = self.dropout(X)
            return X

    class DomainSpecificFeatureModule(torch.nn.Module):
        def __init__(self, num_features=10, nonlin=torch.nn.ReLU()):
            super().__init__()
            self.dense0 = torch.nn.Linear(num_features, num_features)
            self.nonlin = nonlin
            self.dropout = torch.nn.Dropout(0.5)

        def forward(self, X):
            X = self.nonlin(self.dense0(X))
            X = self.dropout(X)
            return X

    class DomainSpecificClassifierModule(torch.nn.Module):
        def __init__(self, num_features=10, n_classes=2):
            super().__init__()
            self.dense1 = torch.nn.Linear(num_features, n_classes)

        def forward(self, X):
            X = self.dense1(X)
            return X

    module_list = [
        CommonFeatureModule(),
        DomainSpecificFeatureModule(),
        DomainSpecificClassifierModule(),
    ]
    domain_specific_layers = [False, True, True]
    module = MultiSourceModule(
        module_list, domain_specific_layers=domain_specific_layers, n_domains=2
    )
    module.eval()

    n_samples = 20
    dataset = make_dataset_from_moons_distribution(
        n_samples_source=n_samples,
        n_samples_target=n_samples,
        pos_source=[0.1, 0.2],
        noise=0.1,
        random_state=42,
        return_dataset=True,
    )
    method = MFSAN(
        module=module,
        reg=1,
        sigmas=sigmas,
        layer_name="output_layer_1",
        batch_size=10,
        max_epochs=10,
        train_split=None,
    )

    X, y, sample_domain = dataset.pack_train(as_sources=["s1", "s0"], as_targets=["t"])
    method.fit(X.astype(np.float32), y.astype(np.int64), sample_domain)

    X_test, y_test, sample_domain_test = dataset.pack_test(as_targets=["t"])

    y_pred = method.predict(X_test.astype(np.float32), sample_domain_test)

    assert y_pred.shape[0] == X_test.shape[0]

    history = method.history_

    assert history[0]["train_loss"] > history[-1]["train_loss"]
