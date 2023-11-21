# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#
# License: BSD 3-Clause

"""
Utilities to produce datasets for testing and benchmarking.
"""

from ._base import (
    DomainAwareDataset,
    get_data_home,
    select_domain,
)
from ._office import (
    Office31CategoriesPreset,
    Office31Domain,
    fetch_office31_decaf,
    fetch_office31_decaf_all,
    fetch_office31_surf,
    fetch_office31_surf_all,
)
from ._samples_generator import (
    make_shifted_blobs,
    make_shifted_datasets,
    make_dataset_from_moons_distribution,
    make_variable_frequency_dataset,
)

__all__ = [
    'DomainAwareDataset',
    'Office31CategoriesPreset',
    'Office31Domain',
    'get_data_home',
    'fetch_office31_decaf',
    'fetch_office31_decaf_all',
    'fetch_office31_surf',
    'fetch_office31_surf_all',
    'make_shifted_blobs',
    'make_shifted_datasets',
    'make_dataset_from_moons_distribution',
    'make_variable_frequency_dataset',
    'select_domain',
]
