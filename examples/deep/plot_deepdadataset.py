"""
Deep Domain Aware Datasets
==========================

This example illustrate some uses of DeepDADatasets.
"""
# Author: Maxence Barneche
#
# License: BSD 3-Clause
# sphinx_gallery_thumbnail_number = 4
# %%

import numpy as np

from skada.datasets import make_shifted_datasets
from skada.deep.base import DeepDADataset as DDAD

# %%
# Creation
# --------
# Deep domain aware datasets are a unified representation of data for deep
# methods in skada.
#
# In those datasets, a data sample has three (optionally, four) attributes:
#   * the data point :code:`X`
#   * the label :code:`y`
#   * the domain :code:`sample_domain`
#   * optionally, the weight :code:`sample_weight`
#
# Note that the data is not shuffled, so the order of the samples is preserved.
#
# .. WARNING::
#   In a dataset, either all data samples have a weight, or none of them.
#   On the other hand, it is possible that a sample has no associated label or domain.
#   In that case, it will be associated to label :code:`-1` and domain :code:`0`.
#
# DeepDADatasets can be created from numpy arrays, torch tensors, lists,
# tuples, pandas dataframes,
# or dictionary of the former.
#
# If a dictionary is provided, it must contain the keys :code:`X`, :code:`y` (optional),
# :code:`sample_domain` (optional) and :code:`sample_weight` (optional).
#
# If both dictionary and positional arguments are provided, the dictionary
# arguments will take precedence over the positional ones.
#
# The data is then stored in torch tensors.

# practice dataset
raw_data = make_shifted_datasets(20, 20, random_state=42)
X, y, sample_domain = raw_data
# though these are not technically weights, they will act as such throughout the guide.
weights = np.ones_like(y)
dict_raw_data = {"X": X, "sample_domain": sample_domain, "y": y}
weighted_dict_raw_data = {
    "X": X,
    "sample_domain": sample_domain,
    "y": y,
    "sample_weight": weights,
}

dataset = DDAD(X, y, sample_domain)
dataset_from_dict = DDAD(dict_raw_data)
# it is possible to add weights to the dataset, either at creation or later
dataset_with_weights = DDAD(X, y, sample_domain, weights)
dataset_with_weights_from_dict = DDAD(weighted_dict_raw_data)

# these methods change the dataset in place and return the dataset itself
dataset.add_weights(weights)
dataset.remove_weights()

# %%
# It is also possible to merge two datasets, which will concatenate the data
# samples, the labels and the domains.
dataset2 = dataset.merge(dataset)

# %%
# Accessing data
# ----------------
#
# The data can be accessed with the same indexing methods as for a torch tensor.
# The returned data is a tuple with a dictionary with the keys :code:`X`,
# :code:`sample_domain`, and optionally :code:`sample_weight` as first element and
# the corresponding label :code:`y` as second element.
#
# ..note::
#   The data is stored in torch tensors, with dimension 1 for :code:`sample_domain`,
#   :code:`sample_weight`, and :code:`y`.
#
# It is also possible to access the data through the various selection methods,
# all of which return DeepDADatasets instances.

dataset[0]  # first sample
dataset[0:5]  # first five samples
dataset.select_domain(1)  # all samples from domain 1
dataset.select(lambda label: label == 1, on="y")  # all samples with label 1
