"""
Visualizing cross-validation behavior in skada
==============================================

This illustrates the use of DA cross-validation object such as 
:class:`~skada.model_selection.RandomShuffleDomainAwareSplit`.
"""  # noqa
# %%
# Let's prepare the imports:

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from skada.model_selection import (
    SourceTargetShuffleSplit,
    LeaveOneDomainOut,
    RandomShuffleDomainAwareSplit,
    GroupDomainAwareKFold,
)
from skada.datasets import make_shifted_datasets

RANDOM_SEED = 0
cmap_data = plt.cm.Paired
cmap_cv = plt.cm.coolwarm
n_splits = 4
# Since we'll be using a dataset with 1 source and 1 target source,
# the lodo splitter will generate only at most 2 splits
n_splits_lodo = 2

# %%
# First we generate our datapoints. They are drawn from two different
# distributions, one source and one target distribution. The target
# distribution is shifted with respect to the source distribution.
# Thus we will have a domain adaptation problem with 1 source domain
# and 1 target domain.

dataset = make_shifted_datasets(
    n_samples_source=3,
    n_samples_target=2,
    shift="concept_drift",
    label="binary",
    noise=0.4,
    random_state=RANDOM_SEED,
    return_dataset=True
)

X, y, sample_domain = dataset.pack_train(as_sources=['s'], as_targets=['t'])
_, target_labels, _ = dataset.pack(as_sources=['s'], as_targets=['t'], train=False)

indx_sort = np.argsort(sample_domain)
X = X[indx_sort]
y = y[indx_sort]
target_labels = target_labels[indx_sort]
sample_domain = sample_domain[indx_sort]

# For Lodo methods
X_lodo, y_lodo, sample_domain_lodo = dataset.pack_lodo()

indx_sort = np.argsort(sample_domain_lodo)
X_lodo = X_lodo[indx_sort]
y_lodo = y_lodo[indx_sort]
sample_domain_lodo = sample_domain_lodo[indx_sort]

# %%
# We’ll define a function that lets us visualize the behavior of
# each cross-validation object.
# We’ll perform 4 splits (or 2 for the lodo method) of the data.
# On each split, we’ll visualize the indices chosen for
# the training set (in blue) and the test set (in red).


# Code source: scikit-learn documentation
# Modified for documentation by Yanis Lalou
# License: BSD 3 clause
def plot_cv_indices(cv, X, y, sample_domain, ax, n_splits, lw=10):
    """Create a sample plot for indices of a cross-validation object."""

    # Generate the training/testing visualizations for each CV split
    if isinstance(cv, GroupDomainAwareKFold):
        cv_args = {'X': X, 'y': y, 'groups': sample_domain}
    else:
        cv_args = {'X': X, 'y': y, 'sample_domain': sample_domain}

    for ii, (tr, tt) in enumerate(cv.split(**cv_args)):
        # Fill in indices with the training/test sample_domain
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2,
        )

    # Plot the data classes and sample_domain at the end
    ax.scatter(
        range(len(X)), [ii + 1.5] * len(X), c=y,
        marker="_", lw=lw, cmap=cmap_data
    )

    ax.scatter(
        range(len(X)), [ii + 2.5] * len(X), c=sample_domain,
        marker="_", lw=lw, cmap=cmap_data
    )

    # Formatting
    yticklabels = list(range(n_splits)) + ["class", "sample_domain"]
    ax.set(
        yticks=np.arange(n_splits + 2) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_splits + 2.2, -0.2],
        xlim=[0, len(X)],
    )
    ax.set_title("{}".format(type(cv).__name__), fontsize=15)
    return ax

# %%
# Let's see how the different cross-validation objects behave on our dataset.


cvs = [SourceTargetShuffleSplit,
       RandomShuffleDomainAwareSplit,
       GroupDomainAwareKFold,
       LeaveOneDomainOut
       ]

for cv in cvs:
    fig, ax = plt.subplots(figsize=(6, 3))

    if cv is LeaveOneDomainOut:
        plot_cv_indices(cv(n_splits_lodo), X_lodo, y_lodo,
                        sample_domain_lodo, ax, n_splits_lodo
                        )
    else:
        plot_cv_indices(cv(n_splits), X, target_labels,
                        sample_domain, ax, n_splits
                        )
    ax.legend(
        [Patch(color=cmap_cv(0.8)), Patch(color=cmap_cv(0.02))],
        ["Testing set", "Training set"],
        loc=(1.02, 0.8),
    )
    # Make the legend fit
    plt.tight_layout()
    fig.subplots_adjust(right=0.7)

# %%
# As we can see each splitter has a very different behavior:
#   -   :class:`~skada.model_selection.SourceTargetShuffleSplit`: Each sample
#       is used once as a test set while the remaining samples
#       form the training set.
#   -   :class:`~skada.model_selection.RandomShuffleDomainAwareSplit`:
#       Randomly split the data depending on their sample_domain.
#       Each fold is composed of samples coming from both
#       source and target domains.
#   -   :class:`~skada.model_selection.GroupDomainAwareKFold`: Same as
#       :class:`~skada.model_selection.RandomShuffleDomainAwareSplit` but the
#       split depends not only on the samples sample_domain but also their label.
#   -   :class:`~skada.model_selection.LeaveOneDomainOut`: Each sample with the same
#       sample_domain is used once as the test set, while the remaining samples
#       form the training set (Can be used only with
#       :func:`~skada.datasets._base.DomainAwareDataset.pack_lodo`)
