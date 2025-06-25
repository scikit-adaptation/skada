# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#
# License: BSD 3-Clause
import pytest

pytest.importorskip("torch")

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from skada import make_da_pipeline
from skada.deep import DANN, DeepCoral
from skada.deep.modules import DomainClassifier, ToyModule2D


@pytest.mark.parametrize(
    "method",
    [
        DANN(
            ToyModule2D(),
            reg=1,
            domain_classifier=DomainClassifier(num_features=10),
            layer_name="dropout",
            batch_size=10,
            max_epochs=10,
            train_split=None,
        ),
        DeepCoral(
            ToyModule2D(),
            reg=1,
            layer_name="dropout",
            batch_size=10,
            max_epochs=10,
            train_split=None,
        ),
    ],
)
def test_pipeline(da_dataset, method):
    module = ToyModule2D()
    module.eval()

    X, y, sample_domain = da_dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )

    pipe = make_da_pipeline(StandardScaler(), method)
    pipe.fit(X.astype(np.float32), y, sample_domain=sample_domain)
