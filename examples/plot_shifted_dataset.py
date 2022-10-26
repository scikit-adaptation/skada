"""
====================================================
Plot dataset source domain and shifted target domain
====================================================
This illustrates the :func:`~skada.datasets.make_shifted_dataset`
dataset generator. Each method consists of generating source data
and shifted target data.
"""

import numpy as np
import matplotlib.pyplot as plt

from skada.datasets import make_shifted_datasets

# Use same random seed for multiple calls to make_shifted_datasets to
# ensure same distributions
RANDOM_SEED = np.random.randint(2**10)


def plot_shifted_dataset(shift, RANDOM_SEED):
    X_source, y_source, X_target, y_target = make_shifted_datasets(
        n_samples_source=20,
        n_samples_target=20,
        shift=shift,
        noise=0.3,
        label='multiclass',
        random_state=RANDOM_SEED
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex="row", sharey="row", figsize=(8, 4))
    fig.suptitle(shift, fontsize=14)
    plt.subplots_adjust(bottom=0.15)
    ax1.scatter(
        X_source[:, 0],
        X_source[:, 1],
        c=y_source,
        cmap='tab10',
        vmax=10,
        alpha=0.5,)
    ax1.set_title("Source data")
    ax1.set_xlabel("Feature 1")
    ax1.set_ylabel("Feature 2")

    ax2.scatter(
        X_source[:, 0],
        X_source[:, 1],
        c=y_source,
        cmap='tab10',
        vmax=10,
        alpha=0.1,)
    ax2.scatter(
        X_target[:, 0],
        X_target[:, 1],
        c=y_target,
        cmap='tab10',
        vmax=10,
        alpha=0.5,)
    ax2.set_title("Target data")
    ax2.set_xlabel("Feature 1")
    ax2.set_ylabel("Feature 2")

    plt.show()


plot_shifted_dataset('covariate_shift', RANDOM_SEED)
plot_shifted_dataset('target_shift', RANDOM_SEED)
plot_shifted_dataset('concept_drift', RANDOM_SEED)
plot_shifted_dataset('sample_bias', RANDOM_SEED)
print("The data was generated from (random_state=%d):" % RANDOM_SEED)

# %%
