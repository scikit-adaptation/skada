"""
Plot dataset source domain and shifted target domain
====================================================

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

from skada import source_target_split
from skada.datasets import make_shifted_datasets

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
        label="regression",
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
        vmax=max(y),
        alpha=0.5,
        cmap="autumn",
    )
    ax1.set_title("Source data")
    ax1.set_xlabel("Feature 1")
    ax1.set_ylabel("Feature 2")

    s = ax2.scatter(
        X_target[:, 0],
        X_target[:, 1],
        c=y_target,
        vmax=max(y),
        alpha=0.5,
        cmap="autumn",
    )
    ax2.set_title("Target data")
    ax2.set_xlabel("Feature 1")
    ax2.set_ylabel("Feature 2")

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.80])
    cb = fig.colorbar(s, cax=cbar_ax)
    cb.set_label("y-value")

    plt.show()


# %% Visualize shifted datasets

for shift in ["covariate_shift", "target_shift", "conditional_shift", "subspace"]:
    plot_shifted_dataset(shift)
