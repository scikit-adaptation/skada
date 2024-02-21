"""
Plot comparison of DA methods
====================================================

A comparison of a several methods of DA in skada on
synthetic datasets. The point of this example is to
illustrate the nature of decision boundaries of
different methods. This should be taken with a grain
of salt, as the intuition conveyed by these examples
does not necessarily carry over to real datasets.


The plots show training points in solid colors then
training points in semi-transparent and testing points
in solid colors. The lower right shows the classification
accuracy on the test set.
"""
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.neighbors import KernelDensity

from skada import (
    ReweightDensity,
    GaussianReweightDensity,
    DiscriminatorReweightDensity,
    KLIEP
)
from skada._reweight import NearestNeighborReweightDensity
from skada import SubspaceAlignment, TransferComponentAnalysis
from skada import (
    OTMapping,
    EntropicOTMapping,
    ClassRegularizerOTMapping,
    LinearOTMapping,
    CORAL
)
from skada.datasets import make_shifted_datasets

# Use same random seed for multiple calls to make_datasets to
# ensure same distributions
RANDOM_SEED = 42

names = [
    "Without da",
    "1NN Reweight Density",
]

classifiers = [
    SVC(),
    NearestNeighborReweightDensity(SVC().set_fit_request(sample_weight=True), laplace_smoothing=True),
]

ns = 10
nt = 10

datasets = [
    make_shifted_datasets(
        n_samples_source=ns,
        n_samples_target=nt,
        shift="covariate_shift",
        label="binary",
        noise=0.4,
        random_state=RANDOM_SEED,
        return_dataset=True
    ),
]

figure, axes = plt.subplots(len(classifiers) + 2, 2, figsize=(9, 27))
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y, sample_domain = ds.pack_train(as_sources=['s'], as_targets=['t'])
    Xs, ys = ds.get_domain("s")
    Xt, yt = ds.get_domain("t")

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    ax = axes[0, ds_cnt]
    if ds_cnt == 0:
        ax.set_ylabel("Source data")
    # Plot the source points
    ax.scatter(
        Xs[:, 0],
        Xs[:, 1],
        c=ys,
        cmap=cm_bright,
        alpha=0.5,
    )

    ax = axes[1, 0]

    if ds_cnt == 0:
        ax.set_ylabel("Target data")
    # Plot the target points
    ax.scatter(
        Xt[:, 0],
        Xt[:, 1],
        c=ys,
        cmap=cm_bright,
        alpha=0.1,
    )
    ax.scatter(
        Xt[:, 0],
        Xt[:, 1],
        c=yt,
        cmap=cm_bright,
        alpha=0.5,
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    i = 2

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        print(name, clf)
        ax = axes[i, ds_cnt]
        if name == "Without da":
            clf.fit(Xs, ys)
        else:
            clf.fit(X, y, sample_domain=sample_domain)
        score = clf.score(Xt, yt)
        DecisionBoundaryDisplay.from_estimator(
            clf, Xs, cmap=cm, alpha=0.8, ax=ax, eps=0.5, response_method="predict",
        )

        if name == "1NN Reweight Density":
            size = 10*clf.named_steps['nearestneighbordensityadapter'].base_estimator.get_weights(Xs, Xt)
        else:
            size = np.array([30]*Xs.shape[0])


        # Plot the target points
        ax.scatter(
            Xt[:, 0],
            Xt[:, 1],
            c=yt,
            cmap=cm_bright,
            alpha=0.5,
        )

        ax.set_xlim(x_min, x_max)

        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_ylabel(name)
        ax.text(
            x_max - 0.3,
            y_min + 0.3,
            ("%.2f" % score).lstrip("0"),
            size=15,
            horizontalalignment="right",
        )

        ax = axes[i, 1]

        # Plot the target points
        ax.scatter(
            Xs[:, 0],
            Xs[:, 1],
            c=ys,
            cmap=cm_bright,
            alpha=0.5,
            s=size
        )

        ax.set_xlim(x_min, x_max)

        ax.set_xticks(())
        ax.set_yticks(())


        i += 1

plt.tight_layout()
plt.show()
