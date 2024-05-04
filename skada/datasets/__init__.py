# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#         Yanis Lalou <yanis.lalou@polytechnique.edu>
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
from ._amazon_review import (
    AmazonReviewDomain,
    fetch_amazon_review,
    fetch_amazon_review_all,
)
from ._office_home import (
    OfficeHomeDomain,
    fetch_office_home,
    fetch_office_home_all,
)
from ._samples_generator import (
    make_shifted_blobs,
    make_shifted_datasets,
    make_dataset_from_moons_distribution,
    make_variable_frequency_dataset,
)
from ._mnist_usps import load_mnist_usps

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
    'load_mnist_usps',
    'fetch_amazon_review',
    'fetch_amazon_review_all',
    'fetch_office_home',
    'fetch_office_home_all',
]
