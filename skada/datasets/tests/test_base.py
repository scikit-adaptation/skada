# Author: Oleksii Kachaiev <kachayev@gmail.com>
#
# License: BSD 3-Clause

import numpy as np
from numpy.testing import assert_array_equal

from skada.datasets import DomainAwareDataset


def test_dataset_train_label_masking():
    dataset = DomainAwareDataset()
    dataset.add_domain(np.array([1., 2.]), np.array([1, 2]), 's1')
    dataset.add_domain(np.array([10., 20., 30.]), np.array([10, 20, 30]), 't1')
    X, y, sample_domain = dataset.pack_for_train(as_sources=['s1'], as_targets=['t1'])

    # test shape of the output
    assert X.shape == (5,)
    assert y.shape == (5,)
    assert sample_domain.shape == (5,)
    assert (sample_domain > 0).sum() == 2
    assert (sample_domain < 0).sum() == 3

    # test label masking
    assert_array_equal(y[sample_domain < 0], np.array([-1, -1, -1]))
    assert np.all(y[sample_domain > 0])

    # custom mask
    X, y, sample_domain = dataset.pack_for_train(
        as_sources=['s1'], as_targets=['t1'], mask=-10)
    assert_array_equal(y[sample_domain < 0], np.array([-10, -10, -10]))
