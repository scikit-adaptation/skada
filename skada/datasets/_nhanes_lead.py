# Author: Yanis Lalou <yanis.lalou@polytechnique.edu>
#
# License: BSD 3-Clause

"""
NHANES-related tools. See also the documentation at the link below:
https://www.cdc.gov/Nchs/Nhanes/about_nhanes.htm

NHANES is a public data source and no special action is required
to access it.

The dataset has been in part preprocessed by the TableShift.
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
        url="""https://raw.githubusercontent.com/YanisLalou/skada_datasets/main/nhanes_lead.csv""",  # noqa: E501
        checksum="59ed30edaa30cf730e6e4ab14f90bceec22d85c4b403953e1297afdb40d670e8",
    )
)


def fetch_nhanes_lead(
    loader_spec: FileLoaderSpec = _NHANES_LEAD_LOADER,
    data_home: Union[None, str, os.PathLike] = None,
    download_if_missing: bool = True,
    return_X_y: bool = False,
    shuffle: bool = False,
    random_state: Union[None, int, np.random.RandomState] = None,
) -> Union[Bunch, Tuple[np.ndarray, np.ndarray]]:
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
        random_state
    )

    return (dataset.X, dataset.y) if return_X_y else dataset


def _load_nhanes_lead(
    dataset_dir: Union[os.PathLike, str],
    shuffle: bool = False,
    random_state: Union[None, int, np.random.RandomState] = None,
) -> Bunch:
    fullpath = Path(dataset_dir)

    # Read the dataset into lists
    nhanes_lead_data = read_dataset(fullpath)

    # Generate the domain-aware dataset
    dataset = generate_domain_aware_dataset(nhanes_lead_data, shuffle, random_state)

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


def generate_domain_aware_dataset(
    nhanes_lead_data,
    shuffle: bool = False,
    random_state: Union[None, int, np.random.RandomState] = None
):
    dataset = DomainAwareDataset()

    # Group the list by the 'INDFMPIRBelowCutoff' column
    grouped_data = {}
    for row in nhanes_lead_data:
        cutoff = row['INDFMPIRBelowCutoff']
        if cutoff not in grouped_data:
            grouped_data[cutoff] = []
        row.pop('INDFMPIRBelowCutoff')
        grouped_data[cutoff].append(row)

    # Populate the dataset with the grouped data
    for domain_name in grouped_data.keys():
        cutoff_data = grouped_data[domain_name]

        target_key = 'LBXBPB'

        cutoff_data_X = remove_key_from_dicts(cutoff_data, target_key)
        X = np.array([list(entry.values()) for entry in cutoff_data_X])
        y = np.array([int(row[target_key]) for row in cutoff_data])

        if shuffle:
            if random_state is None:
                random_state = np.random.RandomState()
            indices = np.arange(X.shape[0])
            random_state.shuffle(indices)
            X = X[indices]
            y = y[indices]

        dataset.add_domain(X, y, domain_name=domain_name)

    return dataset


def remove_key_from_dicts(dicts, target_key):
    dicts_copy = copy.deepcopy(dicts)
    for d in dicts_copy:
        d.pop(target_key)
    return dicts_copy
