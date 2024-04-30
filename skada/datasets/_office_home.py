# Authors: Ambroise Odonnat <ambroiseodonnattechnologie@gmail.com>
#
# License: BSD 3-Clause

"""
OfficeHome dataset features from a pretrained resnet50.
ResNet-50 from `Deep Residual Learning for Image Recognition`.
OfficeHome from `https://github.com/jindongwang/transferlearning/tree/master/data
"""

import ast

from sklearn.datasets._base import RemoteFileMetadata, _fetch_remote
from sklearn.preprocessing import LabelEncoder
from .._utils import _logger

import os
import time
from collections import namedtuple
from ._base import get_data_home, DomainAwareDataset
import numpy as np
from typing import Tuple, Union
from pathlib import Path
from sklearn.utils import Bunch
import csv
import copy

FileLoaderSpec = namedtuple("FileLoaderSpec", ["dataset_dir", "extract_root", "remote"])

_OFFICE_HOME_RESNET50_LOADER = FileLoaderSpec(
    dataset_dir="office_home_resnet50_directory",
    extract_root=False,
    remote=RemoteFileMetadata(
        filename="OfficeHomeResnet50.csv",
        url="""https://figshare.com/ndownloader/files/46002420?private_link=682e4eb7cfef7e179719""",  # noqa: E501
        checksum="9cb6d17d1006047afbe2637736b07b0d59209ec254930af05fffa93150b398f8",
    ),
)


def fetch_domain_aware_office_home_resnet50(
    loader_spec: FileLoaderSpec = _OFFICE_HOME_RESNET50_LOADER,
    data_home: Union[None, str, os.PathLike] = None,
    download_if_missing: bool = True,
    shuffle: bool = False,
    random_state: Union[None, int, np.random.RandomState] = None,
) -> DomainAwareDataset:
    """Load the Resnet 50 features extracted from Office-Home.

    Office-Home is a benchmark dataset for domain adaptation that
    contains 4 domains where each domain consists of 65 categories.
    The four domains are: Art – artistic images in the form of
    sketches, paintings, ornamentation, etc.; Clipart – collection
    of clipart images; Product – images of objects without a background
    and Real-World – images of objects captured with a regular camera.
    It contains 15,500 images, with an average of around 70 images per
    class and a maximum of 99 images in a class.

    For more information, see:
    * https://arxiv.org/pdf/1706.07522v1

    =================   =====================
    Targets                                 65
    Samples total                       15500
    Dimensionality                       2048
    Data Type                           uint8
    =================   =====================

    Parameters
    ----------
    loader_spec : FileLoaderSpec, default=_OFFICE_HOME_RESNET50_LOADER
        The loader specification. It contains the following fields:
        * dataset_dir: str
            The name of the dataset directory.
        * extract_root: bool, default=False
            Whether to extract the dataset in the root of the data folder
            or in a subfolder.
        * remote: RemoteFileMetadata
            The remote file metadata. It contains the following fields:
            * filename: str
                The name of the file.
            * url: str
                The URL of the file.
            * checksum: str
                The SHA256 checksum of the file.
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
    data_home = get_data_home(data_home)
    dataset_dir = os.path.join(data_home, loader_spec.dataset_dir)
    if not os.path.exists(dataset_dir):
        if not download_if_missing:
            raise OSError("Data not found and `download_if_missing` is False")
        os.makedirs(dataset_dir)
        _download_office_home_resnet50(
            loader_spec.remote, data_home if loader_spec.extract_root else dataset_dir
        )

    dataset = _load_office_home_resnet50(
        dataset_dir,
        shuffle,
        random_state,
        return_domain_aware=True,
    )

    return dataset


def fetch_office_home_resnet50(
    loader_spec: FileLoaderSpec = _OFFICE_HOME_RESNET50_LOADER,
    data_home: Union[None, str, os.PathLike] = None,
    download_if_missing: bool = True,
    shuffle: bool = False,
    return_X_y: bool = False,
    random_state: Union[None, int, np.random.RandomState] = None,
) -> Union[Bunch, Tuple[np.ndarray, np.ndarray]]:
    """Load the Resnet 50 features extracted from Office-Home.

    Office-Home is a benchmark dataset for domain adaptation that
    contains 4 domains where each domain consists of 65 categories.
    The four domains are: Art – artistic images in the form of
    sketches, paintings, ornamentation, etc.; Clipart – collection
    of clipart images; Product – images of objects without a background
    and Real-World – images of objects captured with a regular camera.
    It contains 15,500 images, with an average of around 70 images per
    class and a maximum of 99 images in a class.

    For more information, see:
    * https://arxiv.org/pdf/1706.07522v1

    =================   =====================
    Targets                                 65
    Samples total                       15500
    Dimensionality                       2048
    Data Type                           uint8
    =================   =====================

    Parameters
    ----------
    loader_spec : FileLoaderSpec, default=_NHANES_LOADER
        The loader specification. It contains the following fields:
        * dataset_dir: str
            The name of the dataset directory.
        * extract_root: bool, default=False
            Whether to extract the dataset in the root of the data folder
            or in a subfolder.
        * remote: RemoteFileMetadata
            The remote file metadata. It contains the following fields:
            * filename: str
                The name of the file.
            * url: str
                The URL of the file.
            * checksum: str
                The SHA256 checksum of the file.
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

        X: ndarray, shape (n_samples, 16)
            Each row corresponds to a single quantized SURF BoW histogram.
        y : ndarray, shape (n_samples,)
            Labels associated to each image.
        target_names : list
            List of label names for inverse encoding of labels.

    (X, y) : tuple if `return_X_y=True`
        Tuple with the `X` and `y` objects described above.
    """
    data_home = get_data_home(data_home)
    dataset_dir = os.path.join(data_home, loader_spec.dataset_dir)
    if not os.path.exists(dataset_dir):
        if not download_if_missing:
            raise OSError("Data not found and `download_if_missing` is False")
        os.makedirs(dataset_dir)
        _download_office_home_resnet50(
            loader_spec.remote, data_home if loader_spec.extract_root else dataset_dir
        )
    dataset = _load_office_home_resnet50(
        dataset_dir,
        shuffle,
        random_state,
        return_domain_aware=False,
        return_X_y=return_X_y,
    )

    return dataset


def _load_office_home_resnet50(
    dataset_dir: Union[os.PathLike, str],
    shuffle: bool = False,
    random_state: Union[None, int, np.random.RandomState] = None,
    return_domain_aware: bool = False,
    return_X_y: bool = False,
) -> Union[Bunch, Tuple[np.ndarray, np.ndarray], DomainAwareDataset]:

    if return_domain_aware and return_X_y:
        raise ValueError("return_domain_aware and return_X_y cannot both be True")

    fullpath = Path(dataset_dir)

    # Read the dataset into lists
    office_home_resnet50_data = read_dataset(fullpath)

    # Generate the domain-aware dataset
    dataset = generate_dataset(
        office_home_resnet50_data,
        shuffle,
        random_state,
        return_domain_aware,
        return_X_y,
    )

    return dataset


def _download_office_home_resnet50(remote_spec: RemoteFileMetadata, download_dir):
    _logger.info(
        f"Downloading Office-Home-Resnet50 from {remote_spec.url} to {download_dir}"
    )
    started_at = time.time()
    _ = _fetch_remote(remote_spec, dirname=download_dir)
    finished_at = time.time() - started_at
    _logger.info(f"Finished downloading in {finished_at:0.2f} seconds")


def read_dataset(data_folder):
    # Read CSV files into lists
    office_home_resnet50_data = read_csv_with_header(
        os.path.join(data_folder, "OfficeHomeResnet50.csv")
    )

    return office_home_resnet50_data


def read_csv_with_header(file_path):
    data = []
    with open(file_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data


def generate_dataset(
    office_home_resnet50_data,
    shuffle: bool = False,
    random_state: Union[None, int, np.random.RandomState] = None,
    return_domain_aware: bool = True,
    return_X_y: bool = False,
) -> Union[Bunch, Tuple[np.ndarray, np.ndarray], DomainAwareDataset]:
    if return_domain_aware:
        dataset = DomainAwareDataset()
    elif not return_X_y:
        dataset = Bunch()

    target_key = "label"  # Category of object
    domain_key = "domain"  # Type of image (art, product, real-world, product)

    if return_domain_aware:
        # Group the list by the 'domain' column
        grouped_data = {}
        for row in office_home_resnet50_data:
            cutoff = row[domain_key]
            if cutoff not in grouped_data:
                grouped_data[cutoff] = []
            row.pop(domain_key)
            grouped_data[cutoff].append(row)

        # Populate the domain aware dataset with the grouped data
        for domain_name in grouped_data.keys():
            cutoff_data = grouped_data[domain_name]

            cutoff_data_X = remove_key_from_dicts(cutoff_data, target_key)
            X = np.array(
                [ast.literal_eval(entry["features"]) for entry in cutoff_data_X]
            )
            y = np.array([row[target_key] for row in office_home_resnet50_data])

            le = LabelEncoder()
            y = le.fit_transform(y)

            if shuffle:
                shuffle_X_y(X, y, random_state)

            dataset.add_domain(X, y, domain_name=domain_name)

        return dataset
    else:
        cutoff_data_X = remove_key_from_dicts(office_home_resnet50_data, target_key)
        cutoff_data_X = remove_key_from_dicts(cutoff_data_X, domain_key)

        X = np.array([ast.literal_eval(entry["features"]) for entry in cutoff_data_X])
        y = np.array([row[target_key] for row in office_home_resnet50_data])

        le = LabelEncoder()
        y = le.fit_transform(y)

        if shuffle:
            shuffle_X_y(X, y, random_state)

        if not return_X_y:
            dataset.update(
                {
                    "X": X,
                    "y": y,
                }
            )
            return dataset
        else:
            return (X, y)


def remove_key_from_dicts(dicts, target_key):
    dicts_copy = copy.deepcopy(dicts)
    for d in dicts_copy:
        d.pop(target_key)
    return dicts_copy


def shuffle_X_y(X, y, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()
    indices = np.arange(X.shape[0])
    random_state.shuffle(indices)
    X = X[indices]
    y = y[indices]
    return (X, y)
