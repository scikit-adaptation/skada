# Author: Yanis Lalou <yanis.lalou@polytechnique.edu>
#
# License: BSD 3-Clause

from sklearn.datasets._base import RemoteFileMetadata, _fetch_remote
from .._utils import _logger

import os
from zipfile import ZipFile
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

_DIABETES_LOADER = FileLoaderSpec(
    dataset_dir="diabetes_directory",
    extract_root=False,
    remote=RemoteFileMetadata(
        filename="diabetes+130-us+hospitals+for+years+1999-2008.zip",
        url="""https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip""",  # noqa: E501
        checksum="f82ac129da2ddd2299391ff6fbae3a6a58b3edcf59ac9d7bd480c00fe453112a",
    )
)


def fetch_diabetes(
    loader_spec: FileLoaderSpec = _DIABETES_LOADER,
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
        _download_diabetes(
            loader_spec.remote,
            data_home,
            data_home if loader_spec.extract_root else dataset_dir
        )

    dataset = _load_diabetes(
        dataset_dir,
        shuffle,
        random_state
    )

    return (dataset.X, dataset.y) if return_X_y else dataset


def _load_diabetes(
    dataset_dir: Union[os.PathLike, str],
    shuffle: bool = False,
    random_state: Union[None, int, np.random.RandomState] = None,
) -> Bunch:
    fullpath = Path(dataset_dir)

    # Read the dataset into lists
    (admission_type_data, discharge_disposition_data,
        admission_source_data, diabetic_data_data) = read_dataset(fullpath)

    # Preprocess the dataset
    (admission_type_data, discharge_disposition_data,
        admission_source_data, diabetic_data_data) = preprocess_dataset(
        admission_type_data, discharge_disposition_data,
        admission_source_data, diabetic_data_data
            )

    # Generate the domain-aware dataset
    dataset = generate_domain_aware_dataset(diabetic_data_data, shuffle, random_state)

    return dataset


def _download_diabetes(remote_spec: RemoteFileMetadata, download_dir, extract_dir):
    _logger.info(f"Downloading Office31 from {remote_spec.url} to {download_dir}")
    started_at = time.time()
    dataset_path = _fetch_remote(remote_spec, dirname=download_dir)
    finished_at = time.time() - started_at
    _logger.info(f"Finished downloading in {finished_at:0.2f} seconds")
    with ZipFile(dataset_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    os.remove(dataset_path)


def read_dataset(data_folder):
    # Read CSV files into lists
    ids_mapping_data = read_csv(
        os.path.join(data_folder, 'IDS_mapping.csv'), header=False
        )
    diabetic_data_data = read_csv_with_header(
        os.path.join(data_folder, 'diabetic_data.csv')
        )

    # Identify the indices where sections change
    section_indices = [
        i for i, row in enumerate(ids_mapping_data) if all(val == '' for val in row)
    ]
    section_indices = [-1] + section_indices + [len(ids_mapping_data)]

    # Split the lists into sections based on the identified indices
    sections = [
        ids_mapping_data[section_indices[i]+1:section_indices[i+1]]
        for i in range(len(section_indices)-1)
    ]
    sections = [[val for val in section if val is not None] for section in sections]

    # Assign each section to its corresponding list
    admission_type_data, discharge_disposition_data, admission_source_data = sections

    return (admission_type_data, discharge_disposition_data,
            admission_source_data, diabetic_data_data)


def read_csv(file_path, header=True):
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        start_index = 1 if header else 0
        for line in lines[start_index:]:
            data.append(line.strip().split(','))
    return data


def read_csv_with_header(file_path):
    data = []
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data


def preprocess_dataset(
        admission_type_data,
        discharge_disposition_data,
        admission_source_data,
        diabetic_data_data):

    # We don't need these 'encounter_id', 'patient_nbr' columns
    # Not enough non-null values in 'max_glu_serum', 'A1Cresult' columns
    # Drop weight column (97% missing values)
    # Drop medical_specialty column (49% missing values)
    # Drop payer_code column (40% missing values)
    # ------------------------
    # Drop columns diag_1, diag_2, diag_3 (too many categories)
    # + Some are integers, some are floats, some are strings
    # TODO: Clean all these to be integers
    columns_to_drop = ['encounter_id', 'patient_nbr', 'max_glu_serum',
                       'A1Cresult', 'weight', 'payer_code', 'medical_specialty',
                       'diag_1', 'diag_2', 'diag_3']
    diabetic_data_data = [
        {k: v for k, v in entry.items() if k not in columns_to_drop}
        for entry in diabetic_data_data
    ]

    # Drop rows with 'Unknown/Invalid' value (only 3 rows)
    diabetic_data_data = [
        entry for entry in diabetic_data_data if entry['gender'] != 'Unknown/Invalid'
    ]

    # Define a mapping for each age range to its midpoint
    age_mapping = {
        '[70-80)': 75,
        '[60-70)': 65,
        '[50-60)': 55,
        '[80-90)': 85,
        '[40-50)': 45,
        '[30-40)': 35,
        '[90-100)': 95,
        '[20-30)': 25,
        '[10-20)': 15,
        '[0-10)': 5
    }

    columns_to_binary = [
            'diabetesMed', 'change', 'metformin-pioglitazone',
            'metformin-rosiglitazone', 'glimepiride-pioglitazone',
            'glipizide-metformin', 'metformin', 'repaglinide',
            'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide',
            'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
            'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
            'examide', 'citoglipton', 'insulin', 'glyburide-metformin'
        ]

    for entry in diabetic_data_data:
        # Converted to binary (readmit vs. no readmit)
        entry['readmitted'] = 0 if entry['readmitted'] == 'NO' else 1

        # Convert 'gender' to binary
        entry['gender'] = 0 if entry['gender'] == 'Female' else 1

        # Map the 'age' column using the defined mapping
        entry['age'] = age_mapping[entry['age']]

        # Convert columns to binary
        for key in columns_to_binary:
            entry[key] = 0 if entry[key] in ['No', 'Down'] else 1

    return (admission_type_data, discharge_disposition_data,
            admission_source_data, diabetic_data_data)


def generate_domain_aware_dataset(
    diabetic_data_data,
    shuffle: bool = False,
    random_state: Union[None, int, np.random.RandomState] = None
):
    dataset = DomainAwareDataset()

    # Group the list by the 'race' column
    grouped_data = {}
    for row in diabetic_data_data:
        race = row['race']
        if race not in grouped_data:
            grouped_data[race] = []
        row.pop('race')
        grouped_data[race].append(row)

    # Populate the dataset with the grouped data
    for domain_name in grouped_data.keys():
        race_data = grouped_data[domain_name]

        target_key = 'readmitted'

        race_data_X = remove_key_from_dicts(race_data, target_key)
        X = np.array([list(entry.values()) for entry in race_data_X])
        y = np.array([int(row[target_key]) for row in race_data])

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
