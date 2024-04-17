# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#
# License: BSD 3-Clause

import fnmatch
import os
import tarfile
import time
import warnings
from collections import namedtuple
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from scipy.io import loadmat
from sklearn.datasets import load_files
from sklearn.datasets._base import RemoteFileMetadata, _fetch_remote
from sklearn.utils import Bunch

from .._utils import _logger
from ._base import DomainAwareDataset, get_data_home

FileLoaderSpec = namedtuple(
    "FileLoaderSpec",
    [
        "dataset_dir",
        "extract_root",
        "subfolder",
        "filename_pattern",
        "data_key",
        "dtype",
        "dim",
        "remote",
    ],
)

_SURF_LOADER = FileLoaderSpec(
    dataset_dir="domain_adaptation_features",
    extract_root=True,
    subfolder="interest_points",
    filename_pattern="histogram_*.SURF_SURF.amazon_800.SURF_SURF.mat",
    data_key="histogram",
    dtype=np.uint8,
    dim=800,
    remote=RemoteFileMetadata(
        filename="domain_adaptation_features_20110616.tar.gz",
        url="https://figshare.com/ndownloader/files/41786493?private_link=dfe6af3ef4f0f9ae93b9",  # noqa: E501
        checksum="1bb83153343eb0d2c44f66ee63990639176855b2b894fae17ef82c7198123291",
    ),
)

_DECAF_LOADER = FileLoaderSpec(
    dataset_dir="domain_adaptation_decaf_features",
    extract_root=False,
    subfolder="decaf-fts",
    filename_pattern="*",
    data_key="fc8",
    dtype=np.float64,
    dim=1000,
    remote=RemoteFileMetadata(
        filename="domain_adaptation_decaf_features_20140430.tar.gz",
        url="https://figshare.com/ndownloader/files/41786427?private_link=e145dd18e3d010c1f6d9",  # noqa: E501
        checksum="ea13d8ced0fb629937f25f4a4670e0e75fc1955fb439f4ac412e129dd78a19ee",
    ),
)

_CATEGORIES_CALTECH256 = [
    "back_pack",
    "bike",
    "calculator",
    "headphones",
    "keyboard",
    "laptop_computer",
    "monitor",
    "mouse",
    "mug",
    "projector",
]


class Office31Domain(Enum):
    AMAZON = "amazon"
    WEBCAM = "webcam"
    DSLR = "dslr"


class Office31CategoriesPreset(Enum):
    ALL = "all"
    CALTECH256 = "caltech256"


def fetch_office31_surf(
    domain: Union[str, Office31Domain],
    categories: Union[None, Office31CategoriesPreset, List[str]] = None,
    data_home: Union[None, str, os.PathLike] = None,
    download_if_missing: bool = True,
    shuffle: bool = False,
    random_state: Union[None, int, np.random.RandomState] = None,
    return_X_y: bool = False,
) -> Union[Bunch, Tuple[np.ndarray, np.ndarray]]:
    """Load the pre-computed SURF features for Office-31 dataset.

    Office-31 dataset contains 3 domains (Amazon, Webcam, and Dslr).
    Each contains images from amazon.com, or office environment images taken with
    varying lighting and pose changes using a webcam or a dslr camera, respectively.
    Contains 31 categories in each domain.

    SURF BoW histogram features, vector quantized to 800 dimension. The loader
    skips files with 600 dimensions.

    Homepage: https://faculty.cc.gatech.edu/~judy/domainadapt/ (see "Adaptation
    Datasets" section on the page).

    Download it if necessary.

    =================   =====================
    Classes                                31
    Samples total          2813, 795, and 498
    Dimensionality                        800
    Data Type                           uint8
    =================   =====================

    Parameters
    ----------
    domain : str or Office31Domain instance
        Specify which domain to load: 'amazon', 'webcam', or 'dslr'. Note that
        the datasets is fully loaded even when a single domain is requested.

    categories : list or Office31CategoriesPreset instance, default=None
        Specify which categories to load. By default loads all 31 categories
        from the datasets. Commonly used set of 10 categories, so-called 'Caltech-256',
        could be loaded by passing :class:`Office31CategoriesPreset.CALTECH256` value.

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

        X: ndarray, shape (n_samples, 800)
            Each row corresponds to a single quantized SURF BoW histogram.
        y : ndarray, shape (n_samples,)
            Labels associated to each image.
        target_names : list
            List of label names for inverse encoding of labels.

    (X, y) : tuple if `return_X_y=True`
        Tuple with the `X` and `y` objects described above.
    """
    return _fetch_office31(
        _SURF_LOADER,
        domain,
        categories=categories,
        data_home=data_home,
        download_if_missing=download_if_missing,
        shuffle=shuffle,
        random_state=random_state,
        return_X_y=return_X_y,
    )


def fetch_office31_surf_all(
    categories: Union[None, Office31CategoriesPreset, List[str]] = None,
    data_home: Union[None, str, os.PathLike] = None,
    download_if_missing: bool = True,
    shuffle: bool = False,
    random_state: Union[None, int, np.random.RandomState] = None,
) -> DomainAwareDataset:
    """Load all domains for Office-31 SURF dataset.

    Parameters
    ----------
    categories : list or Office31CategoriesPreset instance, default=None
        Specify which categories to load. By default loads all 31 categories
        from the datasets. Commonly used set of 10 categories, so-called 'Caltech-256',
        could be loaded by passing :class:`Office31CategoriesPreset.CALTECH256` value.

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
    return _fetch_office31_all(
        fetch_office31_surf,
        categories=categories,
        data_home=data_home,
        download_if_missing=download_if_missing,
        shuffle=shuffle,
        random_state=random_state,
    )


def fetch_office31_decaf(
    domain: Union[Office31Domain, str],
    categories: Union[None, Office31CategoriesPreset, List[str]] = None,
    data_home: Union[None, str, os.PathLike] = None,
    download_if_missing: bool = True,
    shuffle: bool = False,
    random_state: Union[None, int, np.random.RandomState] = None,
    return_X_y: bool = False,
) -> Union[Bunch, Tuple[np.ndarray, np.ndarray]]:
    """Load the pre-computed DeCAF features for Office-31 dataset.

    Office-31 dataset contains 3 domains (Amazon, Webcam, and Dslr).
    Each contains images from amazon.com, or office environment images taken with
    varying lighting and pose changes using a webcam or a dslr camera, respectively.
    Contains 31 categories in each domain.

    Homepage: https://faculty.cc.gatech.edu/~judy/domainadapt/ (see "Adaptation
    Datasets" section on the page).

    More information on DeCAF: https://proceedings.mlr.press/v32/donahue14.html.

    Download it if necessary.

    =================   =====================
    Classes                                31
    Samples total          2817, 795, and 498
    Dimensionality                       1000
    Data Type           real, between 0 and 1
    =================   =====================

    Parameters
    ----------
    domain : str or Office31Domain instance
        Specify which domain to load: 'amazon', 'webcam', or 'dslr'. Note that
        the datasets is fully loaded even when a single domain is requested.

    categories : list or Office31CategoriesPreset instance, default=None
        Specify which categories to load. By default load all 31 categories from
        the datasets. Commonly used set of 10 categories, so-called 'Caltech-256',
        could be loaded by passing :class:`Office31CategoriesPreset.CALTECH256`.

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

        X: ndarray, shape (n_samples, 1000)
            Each row corresponds to a single decaf vector.
        y : ndarray, shape (n_samples,)
            Labels associated to each image.
        target_names : list
            List of label names for inverse encoding of labels.

    (X, y) : tuple if `return_X_y=True`
        Tuple with the `X` and `y` objects described above.
    """
    return _fetch_office31(
        _DECAF_LOADER,
        domain,
        categories=categories,
        data_home=data_home,
        download_if_missing=download_if_missing,
        shuffle=shuffle,
        random_state=random_state,
        return_X_y=return_X_y,
    )


def fetch_office31_decaf_all(
    categories: Union[None, Office31CategoriesPreset, List[str]] = None,
    data_home: Union[None, str, os.PathLike] = None,
    download_if_missing: bool = True,
    shuffle: bool = False,
    random_state: Union[None, int, np.random.RandomState] = None,
) -> DomainAwareDataset:
    """Load all domains for the Office-31 DeCAF dataset.

    Parameters
    ----------
    categories : list or Office31CategoriesPreset instance, default=None
        Specify which categories to load. By default load all 31 categories from
        the datasets. Commonly used set of 10 categories, so-called 'Caltech-256',
        could be loaded by passing :class:`Office31CategoriesPreset.CALTECH256`.

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
    return _fetch_office31_all(
        fetch_office31_decaf,
        categories=categories,
        data_home=data_home,
        download_if_missing=download_if_missing,
        shuffle=shuffle,
        random_state=random_state,
    )


def _fetch_office31_all(
    loader_fn: Callable,
    categories: Union[None, Office31CategoriesPreset, List[str]] = None,
    data_home: Union[None, str, os.PathLike] = None,
    download_if_missing: bool = True,
    shuffle: bool = False,
    random_state: Union[None, int, np.random.RandomState] = None,
) -> DomainAwareDataset:
    dataset = DomainAwareDataset()
    for domain in Office31Domain:
        X, y = loader_fn(
            domain,
            categories=categories,
            data_home=data_home,
            download_if_missing=download_if_missing,
            shuffle=shuffle,
            random_state=random_state,
            return_X_y=True,
        )
        dataset.add_domain(X, y, domain_name=domain.value)
    return dataset


def _fetch_office31(
    loader_spec: FileLoaderSpec,
    domain: Union[Office31Domain, str],
    categories: Optional[List[str]] = None,
    data_home: Union[None, str, os.PathLike] = None,
    download_if_missing: bool = True,
    shuffle: bool = False,
    random_state: Union[None, int, np.random.RandomState] = None,
    return_X_y: bool = False,
) -> Union[Bunch, Tuple[np.ndarray, np.ndarray]]:
    if not isinstance(domain, Office31Domain):
        domain = Office31Domain(domain.lower())
    data_home = get_data_home(data_home)
    dataset_dir = os.path.join(data_home, loader_spec.dataset_dir)
    if not os.path.exists(dataset_dir):
        if not download_if_missing:
            raise OSError("Data not found and `download_if_missing` is False")
        os.makedirs(dataset_dir)
        # juggling with `extract_root` is only required because SURF features
        # were archived with root folder and DECAF without it
        _download_office31(
            loader_spec.remote,
            data_home,
            data_home if loader_spec.extract_root else dataset_dir,
        )
    if categories == Office31CategoriesPreset.ALL:
        # reset categories filter
        categories = None
    elif categories == Office31CategoriesPreset.CALTECH256:
        categories = _CATEGORIES_CALTECH256
    dataset = _load_office31(
        loader_spec,
        dataset_dir,
        domain,
        categories=categories,
        shuffle=shuffle,
        random_state=random_state,
    )
    return (dataset.X, dataset.y) if return_X_y else dataset


def _download_office31(remote_spec: RemoteFileMetadata, download_dir, extract_dir):
    _logger.info(f"Downloading Office31 from {remote_spec.url} to {download_dir}")
    started_at = time.time()
    dataset_path = _fetch_remote(remote_spec, dirname=download_dir)
    finished_at = time.time() - started_at
    _logger.info(f"Finished downloading in {finished_at:0.2f} seconds")
    tarfile.open(dataset_path, "r:gz").extractall(extract_dir)
    os.remove(dataset_path)


def _load_office31(
    loader_spec: FileLoaderSpec,
    dataset_dir: Union[os.PathLike, str],
    domain: Office31Domain,
    categories: Optional[List[str]] = None,
    shuffle: bool = False,
    random_state: Union[None, int, np.random.RandomState] = None,
) -> Bunch:
    fullpath = Path(dataset_dir) / domain.value / loader_spec.subfolder
    files = load_files(
        fullpath,
        load_content=False,
        categories=categories,
        shuffle=shuffle,
        random_state=random_state,
    )

    if categories is not None:
        not_found = set(categories).difference(set(files["target_names"]))
        if not_found:
            warnings.warn(f"The following categories were not found: {not_found}.")

    data, indices = [], []
    for idx, path in enumerate(files.filenames):
        if fnmatch.fnmatch(Path(path).name, loader_spec.filename_pattern):
            content = np.squeeze(loadmat(path)[loader_spec.data_key])
            assert (
                content.shape[-1] == loader_spec.dim
            ), f"File '{path}' contains array with incorrect dimensions."
            indices.append(idx)
            data.append(content.astype(loader_spec.dtype))
    files["data"] = np.vstack(data)
    files["target"] = files["target"][np.array(indices)]

    files["X"] = files.pop("data")
    files["y"] = files.pop("target")
    return files
