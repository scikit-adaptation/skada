# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Oleksii Kachaiev <kachayev@gmail.com>
#
# License: BSD 3-Clause

import pytest

torch = pytest.importorskip("torch")

import numpy as np

from skada.deep.modules import MNISTtoUSPSNet, ToyCNN, ToyModule2D


def test_toymodule2D():
    module = ToyModule2D()
    module.eval()

    X = torch.tensor(np.random.rand(10, 2), dtype=torch.float32)

    y = module(X)

    assert y.shape[0] == X.shape[0]
    assert y.shape[1] == 2


def test_toycnn():
    module = ToyCNN(
        n_channels=3, input_size=100, n_classes=2, kernel_size=3, out_channels=2
    )
    module.eval()

    X = torch.tensor(np.random.rand(10, 3, 100), dtype=torch.float32)

    y = module(X)

    assert y.shape[0] == X.shape[0]
    assert y.shape[1] == 2


def test_mnist_to_usps_net():
    module = MNISTtoUSPSNet()
    module.eval()

    X = torch.tensor(np.random.rand(10, 1, 28, 28), dtype=torch.float32)

    y = module(X)

    assert y.shape[0] == X.shape[0]
    assert y.shape[1] == 10
