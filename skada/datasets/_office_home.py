# Author: Yanis Lalou <yanis.lalou@polytechnique.edu>
#         Ambroise Odonnat <ambroiseodonnattechnologie@gmail.com>
#
# License: BSD 3-Clause

import os
import time
import zipfile
from collections import namedtuple
from enum import Enum
from pathlib import Path
from typing import Callable, Tuple, Union

import numpy as np
from sklearn.datasets._base import RemoteFileMetadata, _fetch_remote
from sklearn.utils import Bunch

from .._utils import _logger
from ._base import DomainAwareDataset, get_data_home

FileLoaderSpec = namedtuple(
    "FileLoaderSpec",
    [
        "dataset_dir",
        "extract_root",
        "remote",
    ],
)

_OFFICE_HOME_LOADER = FileLoaderSpec(
    dataset_dir="office_home_resnet50_directory",
    extract_root=False,
    remote=RemoteFileMetadata(
        filename="OfficeHomeResnet50.zip",
        url="https://figshare.com/ndownloader/files/46067604?private_link=682e4eb7cfef7e179719",
        checksum="db521d8c4d3dbe2da887ba57b2ce5f43e9f9dc788b8d79479d9ce6398cfd2ff3",
    ),
)


class OfficeHomeDomain(Enum):
    BOOKS = "books"
    DVD = "dvd"
    KITCHEN = "kitchen"
    ELEC = "elec"


def fetch_office_home(
    domain: Union[str, OfficeHomeDomain],
    data_home: Union[None, str, os.PathLike] = None,
    download_if_missing: bool = True,
    shuffle: bool = False,
    random_state: Union[None, int, np.random.RandomState] = None,
    return_X_y: bool = False,
) -> Union[Bunch, Tuple[np.ndarray, np.ndarray]]:
    """Load the Resnet 50 features extracted from Office-Home.

    Office-Home is a benchmark dataset for domain adaptation that
    contains 4 domains where each domain consists of 65 categories.
    The four domains are: Art - artistic images in the form of
    sketches, paintings, ornamentation, etc.; Clipart - collection
    of clipart images; Product - images of objects without a background
    and Real-World - images of objects captured with a regular camera.
    It contains 15,500 images, with an average of around 70 images per
    class and a maximum of 99 images in a class.

    For more information, see:
    * https://arxiv.org/pdf/1706.07522v1

    Download it if necessary.

    =================   =========================
    Classes                                    65
    Samples total                           15500
    Dimensionality                           2048
    Data Type                             float32
    =================   =========================

    Parameters
    ----------
    domain : str or OfficeHomeDomain instance
        Specify which domain to load: 'books', 'dvd', 'kitchen' or 'elec'. Note that
        the datasets is fully loaded even when a single domain is requested.

    data_home : str or path-like, default=None
        Specify another download and cache folder for the datasets. By default
        all skada data is stored in '~/skada_datasets' subfolders.
        See :py:func:`skada.datasets.get_home_folder` for more information.

    download_if_missing : bool, default=True
        If False, raise an OSError if the data is not locally available
        instead of trying to download the data from the source site.

    shuffle : bool, default=False
        If True the order of the dataset is shuffled.

    random_state : int, RandomState instance or None, default=0
        Determines random number generation for dataset shuffling. Pass an int
        for reproducible output across multiple function calls.

    return_X_y : bool, default=False
        If True, returns `(X, y)` instead of a :class:`~sklearn.utils.Bunch`
        object. See below for more information about the `X` and `y` object.

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        X: ndarray, shape (n_samples, 400)
            Each row corresponds to a single review.
        y : ndarray, shape (n_samples,)
            Labels associated to each image.
        target_names : list
            List of label names for inverse encoding of labels.

    (X, y) : tuple if `return_X_y=True`
        Tuple with the `X` and `y` objects described above.
    """
    return _fetch_office_home(
        _OFFICE_HOME_LOADER,
        domain,
        data_home=data_home,
        download_if_missing=download_if_missing,
        shuffle=shuffle,
        random_state=random_state,
        return_X_y=return_X_y,
    )


def fetch_office_home_all(
    data_home: Union[None, str, os.PathLike] = None,
    download_if_missing: bool = True,
    shuffle: bool = False,
    random_state: Union[None, int, np.random.RandomState] = None,
) -> DomainAwareDataset:
    """Load all domains for Amazon review dataset.

    Parameters
    ----------
    data_home : str or path-like, default=None
        Specify another download and cache folder for the datasets. By default
        all skada data is stored in '~/skada_datasets' subfolders.
        See :py:func:`skada.datasets.get_home_folder` for more information.

    download_if_missing : bool, default=True
        If False, raise an OSError if the data is not locally available
        instead of trying to download the data from the source site.

    shuffle : bool, default=False
        If True the order of the dataset is shuffled.

    random_state : int, RandomState instance or None, default=0
        Determines random number generation for dataset shuffling. Pass an int
        for reproducible output across multiple function calls.

    Returns
    -------
    DomainAwareDataset : :class:`~skada.datasets.DomainAwareDataset`
        Container carrying all dataset domains.
    """
    return _fetch_office_home_all(
        fetch_office_home,
        data_home=data_home,
        download_if_missing=download_if_missing,
        shuffle=shuffle,
        random_state=random_state,
    )


def _fetch_office_home_all(
    loader_fn: Callable,
    data_home: Union[None, str, os.PathLike] = None,
    download_if_missing: bool = True,
    shuffle: bool = False,
    random_state: Union[None, int, np.random.RandomState] = None,
) -> DomainAwareDataset:
    dataset = DomainAwareDataset()
    for domain in OfficeHomeDomain:
        X, y = loader_fn(
            domain,
            data_home=data_home,
            download_if_missing=download_if_missing,
            shuffle=shuffle,
            random_state=random_state,
            return_X_y=True,
        )
        dataset.add_domain(X, y, domain_name=domain.value)
    return dataset


def _fetch_office_home(
    loader_spec: FileLoaderSpec,
    domain: Union[OfficeHomeDomain, str],
    data_home: Union[None, str, os.PathLike] = None,
    download_if_missing: bool = True,
    shuffle: bool = False,
    random_state: Union[None, int, np.random.RandomState] = None,
    return_X_y: bool = False,
) -> Union[Bunch, Tuple[np.ndarray, np.ndarray]]:
    if not isinstance(domain, OfficeHomeDomain):
        domain = OfficeHomeDomain(domain.lower())
    data_home = get_data_home(data_home)
    dataset_dir = os.path.join(data_home, loader_spec.dataset_dir)
    if not os.path.exists(dataset_dir):
        if not download_if_missing:
            raise OSError("Data not found and `download_if_missing` is False")
        os.makedirs(dataset_dir)
        # juggling with `extract_root` is only required because SURF features
        # were archived with root folder and DECAF without it
        _download_amazon_review(
            loader_spec.remote,
            data_home,
            data_home if loader_spec.extract_root else dataset_dir,
        )

    (X, y) = _load_amazon_review(
        dataset_dir,
        domain,
        shuffle=shuffle,
        random_state=random_state,
    )

    dataset = DomainAwareDataset()
    dataset.add_domain(X, y, domain.value)

    return (X, y) if return_X_y else dataset


def _download_amazon_review(remote_spec: RemoteFileMetadata, download_dir, extract_dir):
    _logger.info(f"Downloading OfficeHome from {remote_spec.url} to {download_dir}")
    started_at = time.time()
    dataset_path = _fetch_remote(remote_spec, dirname=download_dir)
    finished_at = time.time() - started_at
    _logger.info(f"Finished downloading in {finished_at:0.2f} seconds")

    with zipfile.ZipFile(dataset_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    os.remove(dataset_path)


def _load_amazon_review(
    dataset_dir: Union[os.PathLike, str],
    domain: OfficeHomeDomain,
    shuffle: bool = False,
    random_state: Union[None, int, np.random.RandomState] = None,
) -> Bunch:
    fullpath = Path(dataset_dir)
    csv_path = str(fullpath) + "/" + "OfficeHomeResnet50"

    print(csv_path)
    X = None
    y = None

    if shuffle:
        X, y = shuffle_X_y(X, y, random_state)

    return X, y


def shuffle_X_y(X, y, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()
    indices = np.arange(X.shape[0])
    random_state.shuffle(indices)
    X = X[indices]
    y = y[indices]
    return (X, y)
