# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#         Bueno Ruben <ruben.bueno@polytechnique.edu>
#
# License: BSD 3-Clause

import pytest

try:
    import torchvision
except ImportError:
    torchvision = False

import numpy as np

from skada.datasets import (
    load_mnist_usps,
    DomainAwareDataset,
)
from skada.utils import source_target_split


@pytest.mark.skipif(not torchvision, reason="torchvision is not installed")
@pytest.mark.parametrize("n_classes", [2, 5,],)
def test_make_dataset_from_moons_distribution(n_classes):
    X, y, sample_domain = load_mnist_usps(
        n_classes=n_classes,
        return_X_y=True,
        return_dataset=False,
    )
    X_source, X_target, y_source, y_target = source_target_split(
        X, y, sample_domain=sample_domain
    )

    assert X_source.shape[1:] == (1, 28, 28), "X source shape mismatch"
    assert np.unique(y_source).shape == (n_classes,), "Unexpected number of cluster"
    assert X_target.shape[1:] == (1, 28, 28), "X target shape mismatch"
    assert np.unique(y_target).shape == (n_classes,), "Unexpected number of cluster"

    dataset = load_mnist_usps(
        return_X_y=True,
        return_dataset=True,
    )
    assert isinstance(
        dataset, DomainAwareDataset
    ), "return_dataset=True but a dataset has not been returned"

    X, y, sample_domain = load_mnist_usps(
        n_classes=n_classes,
        return_X_y=True,
        return_dataset=False,
        train=True,
    )
    X_source, X_target, y_source, y_target = source_target_split(
        X, y, sample_domain=sample_domain
    )

    assert X_source.shape[1:] == (1, 28, 28), "X source shape mismatch"
    assert np.unique(y_source).shape == (n_classes,), "Unexpected number of cluster"
    assert X_target.shape[1:] == (1, 28, 28), "X target shape mismatch"
    assert np.unique(y_target).shape == (n_classes,), "Unexpected number of cluster"
