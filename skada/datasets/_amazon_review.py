# Author: Yanis Lalou <yanis.lalou@polytechnique.edu>
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
from scipy.io import loadmat
from sklearn.datasets._base import RemoteFileMetadata, _fetch_remote
from sklearn.utils import Bunch

from .._utils import _logger, _shuffle_arrays
from ._base import DomainAwareDataset, get_data_home

FileLoaderSpec = namedtuple(
    "FileLoaderSpec",
    [
        "dataset_dir",
        "extract_root",
        "remote",
    ],
)

_AMAZON_REVIEW_LOADER = FileLoaderSpec(
    dataset_dir="amazon_review_directory",
    extract_root=False,
    remote=RemoteFileMetadata(
        filename="amazon_review.zip",
        url="https://figshare.com/ndownloader/files/46003500",
        checksum="a37a9702d44edcc5b6c6a8060f6f61b36f32d9127233ee6951eaf691c08ee4fb",
    ),
)


class AmazonReviewDomain(Enum):
    BOOKS = "books"
    DVD = "dvd"
    KITCHEN = "kitchen"
    ELEC = "elec"


def fetch_amazon_review(
    domain: Union[str, AmazonReviewDomain],
    data_home: Union[None, str, os.PathLike] = None,
    download_if_missing: bool = True,
    shuffle: bool = False,
    random_state: Union[None, int, np.random.RandomState] = None,
    return_X_y: bool = False,
) -> Union[Bunch, Tuple[np.ndarray, np.ndarray]]:
    """Load the pre-computed features for Amazon review dataset.

    Amazon review dataset contains 4 domains (Books, DVD, Kitchen and Elec).
    Each contains reviews from amazon.com, with the review text and the
    corresponding boolean star rating (Good or Bad).
    It contains 2 categories in each domain.

    Each features has 400 dimension.

    References
    ----------
    J. McAuley, C. Targett, Q. Shi, A. Van Den Hangel, Image-based
    recommendations on styles and substitutes, SIGIR 2015

    Download it if necessary.

    =================   =========================
    Classes                                     2
    Samples total       2000, 1999, 1999 and 1998
    Dimensionality                            400
    Data Type                               uint8
    =================   =========================

    Parameters
    ----------
    domain : str or AmazonReviewDomain instance
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
    return _fetch_amazon_review(
        _AMAZON_REVIEW_LOADER,
        domain,
        data_home=data_home,
        download_if_missing=download_if_missing,
        shuffle=shuffle,
        random_state=random_state,
        return_X_y=return_X_y,
    )


def fetch_amazon_review_all(
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
    return _fetch_amazon_review_all(
        fetch_amazon_review,
        data_home=data_home,
        download_if_missing=download_if_missing,
        shuffle=shuffle,
        random_state=random_state,
    )


def _fetch_amazon_review_all(
    loader_fn: Callable,
    data_home: Union[None, str, os.PathLike] = None,
    download_if_missing: bool = True,
    shuffle: bool = False,
    random_state: Union[None, int, np.random.RandomState] = None,
) -> DomainAwareDataset:
    dataset = DomainAwareDataset()
    for domain in AmazonReviewDomain:
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


def _fetch_amazon_review(
    loader_spec: FileLoaderSpec,
    domain: Union[AmazonReviewDomain, str],
    data_home: Union[None, str, os.PathLike] = None,
    download_if_missing: bool = True,
    shuffle: bool = False,
    random_state: Union[None, int, np.random.RandomState] = None,
    return_X_y: bool = False,
) -> Union[Bunch, Tuple[np.ndarray, np.ndarray]]:
    if not isinstance(domain, AmazonReviewDomain):
        domain = AmazonReviewDomain(domain.lower())
    data_home = get_data_home(data_home)
    dataset_dir = os.path.join(data_home, loader_spec.dataset_dir)
    if not os.path.exists(dataset_dir):
        if not download_if_missing:
            raise OSError("Data not found and `download_if_missing` is False")
        os.makedirs(dataset_dir)
        _download_amazon_review(
            loader_spec.remote,
            data_home,
            dataset_dir,
        )

    (X, y) = _load_amazon_review(
        dataset_dir,
        domain,
        shuffle=shuffle,
        random_state=random_state,
    )

    if return_X_y:
        return (X, y)
    else:
        dataset = DomainAwareDataset()
        dataset.add_domain(X, y, domain.value)
        return dataset


def _download_amazon_review(remote_spec: RemoteFileMetadata, download_dir, extract_dir):
    _logger.info(f"Downloading AmazonReview from {remote_spec.url} to {download_dir}")
    started_at = time.time()
    dataset_path = _fetch_remote(remote_spec, dirname=download_dir)
    finished_at = time.time() - started_at
    _logger.info(f"Finished downloading in {finished_at:0.2f} seconds")

    with zipfile.ZipFile(dataset_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    extracted_folder = str(extract_dir) + "/" + "amazon_review"
    files_to_move = os.listdir(extracted_folder)

    # Move each file to the destination folder
    for file_name in files_to_move:
        # Construct the full path of the file
        source_file_path = os.path.join(extracted_folder, file_name)
        destination_file_path = os.path.join(extract_dir, file_name)

        # Move the file
        os.rename(source_file_path, destination_file_path)

    # Delete the source folder
    os.rmdir(extracted_folder)
    os.remove(dataset_path)


def _load_amazon_review(
    dataset_dir: Union[os.PathLike, str],
    domain: AmazonReviewDomain,
    shuffle: bool = False,
    random_state: Union[None, int, np.random.RandomState] = None,
) -> Bunch:
    fullpath = Path(dataset_dir)

    mat_file = domain.value + "_400.mat"
    mat = loadmat(fullpath / mat_file)

    X = np.array(
        mat["fts"],
        dtype=np.float32,
    )

    y = np.array(
        mat["labels"].flatten(),
        dtype=np.float32,
    )

    if shuffle:
        X, y = _shuffle_arrays(X, y, random_state=random_state)

    return X, y
