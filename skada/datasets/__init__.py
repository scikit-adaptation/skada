"""
Utilities to produce datasets for testing and benchmarking.
"""

from ._samples_generator import (
    make_shifted_blobs,
    make_shifted_datasets,
    make_dataset_from_moons_distribution,
    make_variable_frequency_dataset
)

__all__ = [
    'make_shifted_blobs',
    'make_shifted_datasets',
    'make_dataset_from_moons_distribution',
    'make_variable_frequency_dataset'
]
