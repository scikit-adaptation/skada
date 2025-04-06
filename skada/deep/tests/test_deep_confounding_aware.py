# Author: Calvin McCarter <mccarter.calvin@gmail.com>
#
# License: BSD 3-Clause
import pytest

torch = pytest.importorskip("torch")

import numpy as np

from skada.deep import ConDoAdapterMMD


def test_condo_adapter_mmd():
    # XXX remove this after making true wrapper
    condoer = ConDoAdapterMMD(n_epochs=100)
    X_T = np.sort(np.random.uniform(0, 8, size=(100, 1)))
    X_S = np.sort(np.random.uniform(4, 8, size=(100, 1)))
    Y_T = np.random.normal(4 * X_T + 1, 1 * X_T + 1)
    Y_Strue = np.random.normal(4 * X_S + 1, 1 * X_S + 1)
    Y_S = 5 * Y_Strue + 2
    condoer.fit(Y_S, Y_T, X_S, X_T)
    Y_S2T = condoer.transform(Y_S)
    after_err = np.sqrt(np.mean((Y_S2T - Y_Strue) ** 2))
    assert after_err < 10.0
