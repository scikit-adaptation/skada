# Author: Yanis Lalou <yanis.lalou@polytechnique.edu>
#
# License: BSD 3-Clause

"""
NHANES-related tools. See also the documentation at the link below:
https://www.cdc.gov/Nchs/Nhanes/about_nhanes.htm

NHANES is a public data source and no special action is required
to access it.

The dataset has been in part preprocessed by TableShift.
For more information the preprocess part from TableShift, see:
* https://tableshift.org/datasets.html
* https://github.com/mlfoundations/tableshift

"""

from sklearn.datasets._base import RemoteFileMetadata, _fetch_remote
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

FileLoaderSpec = namedtuple(
    "FileLoaderSpec",
    [
        "dataset_dir",
        "extract_root",
        "remote"
    ]
)

_NHANES_LEAD_LOADER = FileLoaderSpec(
    dataset_dir="nhanes_directory",
    extract_root=False,
    remote=RemoteFileMetadata(
        filename="nhanes_lead.csv",
        url="""https://figshare.com/ndownloader/files/44212007""",  # noqa: E501
        checksum="59ed30edaa30cf730e6e4ab14f90bceec22d85c4b403953e1297afdb40d670e8",
    )
)


def fetch_domain_aware_nhanes_lead(
    loader_spec: FileLoaderSpec = _NHANES_LEAD_LOADER,
    data_home: Union[None, str, os.PathLike] = None,
    download_if_missing: bool = True,
    shuffle: bool = False,
    random_state: Union[None, int, np.random.RandomState] = None,
) -> DomainAwareDataset:
    """Load the Childhood Lead dataset aka nhanes_lead dataset.

    The nhances_lead dataset contains 2 domains: above_PIR and below_PIR.
    It contains 1 binary target label: whether the patient has a blood lead
    level (BLL) above CDC Blood Level Reference Value of 3.5 µg/dL of blood.

    It is composed of Geographic, Demographic, and Health data from the
    National Health and Nutrition Examination Survey (NHANES) 1999-2018.

    NHANES-related dataset. See also the documentation at the link below:
    https://www.cdc.gov/Nchs/Nhanes/about_nhanes.htm

    NHANES is a public data source and no special action is required
    to access it.

    The dataset has been in part preprocessed by TableShift.
    For more information the preprocess part from TableShift, see:
    * https://tableshift.org/datasets.html
    * https://github.com/mlfoundations/tableshift

    Download it if necessary.

    =================   =====================
    Targets                                 2
    Samples total         27499 (14759, 12740)
    Dimensionality                         16
    Data Type                           uint8
    =================   =====================

    Parameters
    ----------
    loader_spec : FileLoaderSpec, default=_NHANES_LEAD_LOADER
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
        _download_nhanes_lead(
            loader_spec.remote,
            data_home if loader_spec.extract_root else dataset_dir
        )

    dataset = _load_nhanes_lead(
        dataset_dir,
        shuffle,
        random_state,
        return_domain_aware=True,
    )

    return dataset


def fetch_nhanes_lead(
    loader_spec: FileLoaderSpec = _NHANES_LEAD_LOADER,
    data_home: Union[None, str, os.PathLike] = None,
    download_if_missing: bool = True,
    shuffle: bool = False,
    return_X_y: bool = False,
    random_state: Union[None, int, np.random.RandomState] = None,
) -> Union[Bunch, Tuple[np.ndarray, np.ndarray]]:
    """Load the Childhood Lead dataset aka nhanes_lead dataset.

    The nhances_lead dataset contains 2 domains: above_PIR and below_PIR.
    It contains 1 binary target label: whether the patient has a blood lead
    level (BLL) above CDC Blood Level Reference Value of 3.5 µg/dL of blood.

    It is composed of Geographic, Demographic, and Health data from the
    National Health and Nutrition Examination Survey (NHANES) 1999-2018.

    NHANES-related dataset. See also the documentation at the link below:
    https://www.cdc.gov/Nchs/Nhanes/about_nhanes.htm

    NHANES is a public data source and no special action is required
    to access it.

    The dataset has been in part preprocessed by TableShift.
    For more information the preprocess part from TableShift, see:
    * https://tableshift.org/datasets.html
    * https://github.com/mlfoundations/tableshift

    Download it if necessary.

    =================   =====================
    Targets                                 2
    Samples total         27499 (14759, 12740)
    Dimensionality                         16
    Data Type                           uint8
    =================   =====================

    Parameters
    ----------
    loader_spec : FileLoaderSpec, default=_NHANES_LEAD_LOADER
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
        _download_nhanes_lead(
            loader_spec.remote,
            data_home if loader_spec.extract_root else dataset_dir
        )

    dataset = _load_nhanes_lead(
        dataset_dir,
        shuffle,
        random_state,
        return_domain_aware=False,
        return_X_y=return_X_y
    )

    return dataset


def _load_nhanes_lead(
    dataset_dir: Union[os.PathLike, str],
    shuffle: bool = False,
    random_state: Union[None, int, np.random.RandomState] = None,
    return_domain_aware: bool = False,
    return_X_y: bool = False,
) -> Union[Bunch, Tuple[np.ndarray, np.ndarray], DomainAwareDataset]:

    if return_domain_aware and return_X_y:
        raise ValueError(
            "return_domain_aware and return_X_y cannot both be True"
        )

    fullpath = Path(dataset_dir)

    # Read the dataset into lists
    nhanes_lead_data = read_dataset(fullpath)

    # Generate the domain-aware dataset
    dataset = generate_dataset(
        nhanes_lead_data, shuffle, random_state, return_domain_aware, return_X_y
    )

    return dataset


def _download_nhanes_lead(remote_spec: RemoteFileMetadata, download_dir):
    _logger.info(f"Downloading Nhanes_lead from {remote_spec.url} to {download_dir}")
    started_at = time.time()
    _ = _fetch_remote(remote_spec, dirname=download_dir)
    finished_at = time.time() - started_at
    _logger.info(f"Finished downloading in {finished_at:0.2f} seconds")


def read_dataset(data_folder):
    # Read CSV files into lists
    nhanes_lead_data = read_csv_with_header(
        os.path.join(data_folder, 'nhanes_lead.csv')
        )

    return nhanes_lead_data


def read_csv_with_header(file_path):
    data = []
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data


def generate_dataset(
    nhanes_lead_data,
    shuffle: bool = False,
    random_state: Union[None, int, np.random.RandomState] = None,
    return_domain_aware: bool = True,
    return_X_y: bool = False,
) -> Union[Bunch, Tuple[np.ndarray, np.ndarray], DomainAwareDataset]:
    if return_domain_aware:
        dataset = DomainAwareDataset()
    elif not return_X_y:
        dataset = Bunch()

    target_key = 'LBXBPB'  # Blood lead level
    domain_key = 'INDFMPIRBelowCutoff'  # Income to poverty ratio

    if return_domain_aware:
        # Group the list by the 'INDFMPIRBelowCutoff' column
        grouped_data = {}
        for row in nhanes_lead_data:
            cutoff = row[domain_key]
            if cutoff not in grouped_data:
                grouped_data[cutoff] = []
            row.pop(domain_key)
            grouped_data[cutoff].append(row)

        # Populate the domain aware dataset with the grouped data
        for domain_name in grouped_data.keys():
            cutoff_data = grouped_data[domain_name]

            cutoff_data_X = remove_key_from_dicts(cutoff_data, target_key)
            X = np.array([list(entry.values()) for entry in cutoff_data_X])
            y = np.array([int(row[target_key]) for row in cutoff_data])

            if shuffle:
                shuffle_X_y(X, y, random_state)

            dataset.add_domain(X, y, domain_name=domain_name)

        return dataset
    else:
        cutoff_data_X = remove_key_from_dicts(nhanes_lead_data, target_key)
        cutoff_data_X = remove_key_from_dicts(cutoff_data_X, domain_key)

        X = np.array([list(entry.values()) for entry in cutoff_data_X])
        y = np.array([int(row[target_key]) for row in nhanes_lead_data])

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
