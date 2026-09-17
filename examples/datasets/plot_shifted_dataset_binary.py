"""
Shifted binary classification dataset
============================================

This illustrates the :func:`~skada.datasets.make_shifted_dataset`
dataset generator. Each method consists of generating source data
and shifted target data. We illustrate here:
covariate shift, target shift, conditional shift, and sample bias.
See detailed description of each shift in [1]_.

.. [1] Moreno-Torres, J. G., Raeder, T., Alaiz-Rodriguez,
       R., Chawla, N. V., and Herrera, F. (2012).
       A unifying view on dataset shift in classification.
       Pattern recognition, 45(1):521-530.
"""

# Author: Ruben Bueno <ruben.bueno@polytechnique.edu>
#
# License: BSD 3-Clause

# %% Imports

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from skada import source_target_split
from skada.datasets import make_shifted_datasets

# Binary classification: color the two classes like the SKADA logo.
cmap_binary = ListedColormap(["#c84630", "#2364aa"])

# %% Helper function


def plot_shifted_dataset(shift, random_state=42):
    """Plot source and shifted target data for a given type of shift.

    The possible shifts are 'covariate_shift', 'target_shift',
    'conditional_shift' or 'subspace'.

    We use here the same random seed for multiple calls to
    ensure same distributions.
    """
    X, y, sample_domain = make_shifted_datasets(
        n_samples_source=20,
        n_samples_target=20,
        shift=shift,
        noise=0.3,
        label="binary",
        random_state=random_state,
    )
    X_source, X_target, y_source, y_target = source_target_split(
        X, y, sample_domain=sample_domain
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex="row", sharey="row", figsize=(8, 4))
    fig.suptitle(shift.replace("_", " ").title(), fontsize=14)
    plt.subplots_adjust(bottom=0.15)
    ax1.scatter(
        X_source[:, 0],
        X_source[:, 1],
        c=y_source,
        cmap=cmap_binary,
        alpha=0.5,
    )
    ax1.set_title("Source data")
    ax1.set_xlabel("Feature 1")
    ax1.set_ylabel("Feature 2")

    ax2.scatter(
        X_target[:, 0],
        X_target[:, 1],
        c=y_target,
        cmap=cmap_binary,
        alpha=0.5,
    )
    ax2.set_title("Target data")
    ax2.set_xlabel("Feature 1")
    ax2.set_ylabel("Feature 2")

    plt.show()


# %% Visualize shifted datasets

for shift in ["covariate_shift", "target_shift", "conditional_shift", "subspace"]:
    plot_shifted_dataset(shift)
