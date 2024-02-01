# Author: Yanis Lalou <yanis.lalou@polytechnique.edu>
#
# License: BSD 3-Clause

import numpy as np
from typing import Tuple
import pytest
import random

from sklearn.datasets._base import RemoteFileMetadata
from sklearn.utils import Bunch
from skada.datasets import fetch_nhanes_lead, fetch_domain_aware_nhanes_lead
from skada.datasets._nhanes_lead import FileLoaderSpec, _load_nhanes_lead
from skada.datasets import DomainAwareDataset

RANDOM_NUMBER = random.random()


def test_fetch_nhanes_lead():
    domain_aware_dataset = fetch_domain_aware_nhanes_lead()
    assert isinstance(domain_aware_dataset, DomainAwareDataset)

    dataset = fetch_nhanes_lead(return_X_y=False)
    assert isinstance(dataset, Bunch)

    dataset = fetch_nhanes_lead(return_X_y=True)
    assert (
        isinstance(dataset, Tuple) and
        isinstance(dataset[0], np.ndarray) and
        isinstance(dataset[1], np.ndarray)
    )

    with pytest.raises(OSError):
        fetch_nhanes_lead(data_home=str(RANDOM_NUMBER),
                          download_if_missing=False)

    with pytest.raises(OSError):
        fetch_domain_aware_nhanes_lead(data_home=str(RANDOM_NUMBER),
                                       download_if_missing=False)


def test_fetch_nhanes_wrong_url():
    _nhanes_lead_loader = FileLoaderSpec(
        dataset_dir=f"dont_exist_directory_{RANDOM_NUMBER}",
        extract_root=False,
        remote=RemoteFileMetadata(
            filename="nhanes_lead.csv",
            url=f"https://wrong_url_{RANDOM_NUMBER}",  # noqa: E501
            checksum="59ed30edaa30cf730e6e4ab14f90bceec22d85c4b403953e1297afdb40d670e8",
        )
    )

    with pytest.raises(Exception):
        fetch_nhanes_lead(
            loader_spec=_nhanes_lead_loader
        )


def test_fetch_nhanes_no_download():
    _nhanes_lead_loader = FileLoaderSpec(
        dataset_dir=f"dont_exist_directory_{RANDOM_NUMBER}",
        extract_root=False,
        remote=RemoteFileMetadata(
            filename="nhanes_lead.csv",
            url="""https://figshare.com/ndownloader/files/44212007""",  # noqa: E501
            checksum="59ed30edaa30cf730e6e4ab14f90bceec22d85c4b403953e1297afdb40d670e8",
        )
    )

    with pytest.raises(OSError):
        fetch_nhanes_lead(
            loader_spec=_nhanes_lead_loader,
            download_if_missing=False
        )


def test_domain_aware_and_return_X_y():
    with pytest.raises(ValueError):
        _load_nhanes_lead(
            dataset_dir="nhanes_directory",
            shuffle=False,
            random_state=None,
            return_domain_aware=True,
            return_X_y=True
        )


def test_shuffle_nhanes_dataset():
    domain_dataset = fetch_domain_aware_nhanes_lead()
    domain_dataset_shuffle = fetch_domain_aware_nhanes_lead(shuffle=True)

    assert ~np.array_equal(domain_dataset.domains_, domain_dataset_shuffle.domains_)

    dataset = fetch_nhanes_lead(return_X_y=False)
    dataset_shuffle = fetch_nhanes_lead(return_X_y=False, shuffle=True)

    assert ~np.array_equal(dataset.X, dataset_shuffle.X)
    assert ~np.array_equal(dataset.y, dataset_shuffle.y)
