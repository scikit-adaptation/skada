# Author: Yanis Lalou <yanis.lalou@polytechnique.edu>
#
# License: BSD 3-Clause

import numpy as np
from typing import Tuple
import pytest
import random

from sklearn.datasets._base import RemoteFileMetadata
from sklearn.utils import Bunch
from skada.datasets import (
    fetch_office_home_resnet50,
    fetch_domain_aware_office_home_resnet50,
)
from skada.datasets._office_home import FileLoaderSpec, _load_office_home_resnet50
from skada.datasets import DomainAwareDataset

RANDOM_NUMBER = random.random()


def test_fetch_office_home_resnet50():
    domain_aware_dataset = fetch_domain_aware_office_home_resnet50()
    assert isinstance(domain_aware_dataset, DomainAwareDataset)

    dataset = fetch_office_home_resnet50(return_X_y=False)
    assert isinstance(dataset, Bunch)

    dataset = fetch_office_home_resnet50(return_X_y=True)
    assert (
        isinstance(dataset, Tuple)
        and isinstance(dataset[0], np.ndarray)
        and isinstance(dataset[1], np.ndarray)
    )

    with pytest.raises(OSError):
        fetch_office_home_resnet50(
            data_home=str(RANDOM_NUMBER), download_if_missing=False
        )

    with pytest.raises(OSError):
        fetch_domain_aware_office_home_resnet50(
            data_home=str(RANDOM_NUMBER), download_if_missing=False
        )


def test_fetch_office_home_resnet50_wrong_url():
    _office_home_resnet50_loader = FileLoaderSpec(
        dataset_dir=f"dont_exist_directory_{RANDOM_NUMBER}",
        extract_root=False,
        remote=RemoteFileMetadata(
            filename="OfficeHomeResnet50.csv",
            url=f"https://wrong_url_{RANDOM_NUMBER}",  # noqa: E501
            checksum="9cb6d17d1006047afbe2637736b07b0d59209ec254930af05fffa93150b398f8",
        ),
    )

    with pytest.raises(Exception):
        fetch_office_home_resnet50(loader_spec=_office_home_resnet50_loader)


def test_office_home_resnet50_no_download():
    _office_home_resnet50_loader = FileLoaderSpec(
        dataset_dir=f"dont_exist_directory_{RANDOM_NUMBER}",
        extract_root=False,
        remote=RemoteFileMetadata(
            filename="OfficeHomeResnet50.csv",
            url="""https://figshare.com/s/682e4eb7cfef7e179719""",  # noqa: E501
            checksum="9cb6d17d1006047afbe2637736b07b0d59209ec254930af05fffa93150b398f8",
        ),
    )

    with pytest.raises(OSError):
        fetch_office_home_resnet50(
            loader_spec=_office_home_resnet50_loader, download_if_missing=False
        )


def test_domain_aware_and_return_X_y():
    with pytest.raises(ValueError):
        _load_office_home_resnet50(
            dataset_dir="office_home_resnet50_directory",
            shuffle=False,
            random_state=None,
            return_domain_aware=True,
            return_X_y=True,
        )


def test_shuffle_office_home_resnet50_dataset():
    domain_dataset = fetch_domain_aware_office_home_resnet50()
    domain_dataset_shuffle = fetch_domain_aware_office_home_resnet50(shuffle=True)

    assert ~np.array_equal(domain_dataset.domains_, domain_dataset_shuffle.domains_)

    dataset, (X, y) = fetch_office_home_resnet50(return_X_y=True)
    dataset_shuffle, (X_shuffle, y_shuffle) = fetch_office_home_resnet50(
        return_X_y=True, shuffle=True
    )

    assert ~np.array_equal(X, X_shuffle)
    assert ~np.array_equal(y, y_shuffle)
