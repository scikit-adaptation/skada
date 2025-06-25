# Author: Tom Mariani <tom@yneuro.com>
#
# License: BSD 3-Clause
import pytest

pytest.importorskip("torch")

import numpy as np
import torch

from skada.datasets import DomainAwareDataset
from skada.deep.base import DeepDADataset


def test_dataset_return_types():
    dataset = DomainAwareDataset()
    dataset.add_domain(torch.Tensor([1.0, 2.0]), torch.Tensor([1, 2]), "s1")
    dataset.add_domain(
        torch.Tensor([10.0, 20.0, 30.0]), torch.Tensor([10, 20, 30]), "t1"
    )
    X, y, sample_domain = dataset.pack(
        as_sources=["s1"], as_targets=["t1"], mask_target_labels=False
    )

    # test that the output is a torch tensor
    assert isinstance(X, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert isinstance(sample_domain, torch.Tensor)

    deep_dataset = dataset.pack(
        as_sources=["s1"],
        as_targets=["t1"],
        mask_target_labels=False,
        return_type="DeepDADataset",
    )
    assert isinstance(deep_dataset, DeepDADataset)

    X, y, sample_domain = dataset.pack(
        as_sources=["s1"],
        as_targets=["t1"],
        mask_target_labels=False,
        return_type="array",
    )

    # test that the output is a numpy array
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(sample_domain, np.ndarray)
