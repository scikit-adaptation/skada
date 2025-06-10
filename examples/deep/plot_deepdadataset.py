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
import pandas as pd
import torch

from skada.datasets import make_shifted_datasets
from skada.deep.base import DeepDADataset

# %%
# Creation
# --------
# Deep domain aware datasets are a unified representation of data for deep
# methods in skada.
#
# In those datasets, a data sample has four (optionally, five) attributes:
#   * the data point :code:`X`
#   * the label :code:`y`
#   * the domain :code:`sample_domain`
#   * optionally, the weight :code:`sample_weight`
#   * the sample index :code:`sample_idx` (automatically generated), which is
#       the index of the sample in the dataset, relative to its domain.
#
# Note that the data is not shuffled, so the order of the samples is preserved.
#
# .. WARNING::
#   In a dataset, either all data samples have a weight, or none of them.
#   On the other hand, it is possible that a sample has no associated label or domain.
#   In that case, it will be associated to label :code:`-1` and domain :code:`0`.
#
# DeepDADatasets can be created from numpy arrays, torch tensors, lists,
# tuples, or dictionary of one of the former.
#
# If a dictionary is provided, it must contain the keys :code:`X`, :code:`y`(optional),
# :code:`sample_domain`(optional) and :code:`sample_weight`(optional).
#
# If both dictionary and positional arguments are provided, the dictionary
# arguments will take precedence over the positional ones.

# practice dataset as numpy arrays
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

dataset = DeepDADataset(X, y, sample_domain)
dataset_from_dict = DeepDADataset(dict_raw_data)
# it is possible to add weights to the dataset, either at creation or later
dataset_with_weights = DeepDADataset(X, y, sample_domain, weights)
dataset_with_weights_from_dict = DeepDADataset(weighted_dict_raw_data)

# these methods change the dataset in place and return the dataset itself
dataset = dataset.add_weights(weights)
dataset = dataset.remove_weights()

# %%
# It is also possible to create a DeepDADataset from lists, tuples, tensors,
# pandas dataframes or any combination of those.
#
# .. note::
#   Just like for the dictionary, if a pandas dataframe is provided it must
#   contain the keys :code:`X`, :code:`y` (optional), :code:`sample_domain`(optional)
#   and :code:`sample_weight` (optional).
#   Also, the data in the dataframe will take precedence over the positional arguments.

# from lists
dataset_from_list = DeepDADataset(X.tolist(), y.tolist(), sample_domain.tolist())
# from tuples
dataset_from_tuple = DeepDADataset(
    tuple(X.tolist()), tuple(y.tolist()), tuple(sample_domain.tolist())
)

# from torch tensors
dataset_from_tensor = DeepDADataset(
    torch.tensor(X), torch.tensor(y), torch.tensor(sample_domain)
)

# from pandas dataframe of same structure as the dictionary
df = pd.DataFrame(
    {
        "X": list(X),
        "y": y,
        "sample_domain": sample_domain,
        "sample_weight": weights,
    }
)
dataset_from_df = DeepDADataset(df)

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
# :code:`sample_domain`, :code:`sample_idx`, and optionally :code:`sample_weight`
# as first element and the corresponding label :code:`y` as second element.
#
# ..note::
#   The data is stored in torch tensors, with dimension 1 for :code:`sample_domain`,
#   :code:`y` and :code:`sample_weight`.
#
# It is also possible to access the data through the various selection methods,
# all of which return DeepDADatasets instances.

# indexing methods return a tuple with the data as dict and the label
first_sample = dataset[0]  # first sample
first_five_samples = dataset[0:5]  # first five samples

# selecting methods return a DeepDADataset with the selected samples
domain_1_samples = dataset.select_domain(1)  # all samples from domain 1
label_1_samples = dataset.select(
    lambda label: label == 1, on="y"
)  # all samples with label 1
