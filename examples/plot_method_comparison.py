"""
Plot comparison of DA methods
====================================================

A comparison of a several methods of DA in skada on
synthetic datasets. The point of this example is to
illustrate the nature of decision boundaries of
different methods. This should be taken with a grain
of salt, as the intuition conveyed by these examples
does not necessarily carry over to real datasets.


The plots show training points in solid colors and
testing points semi-transparent. The lower right
shows the classification accuracy on the test set.
"""
# %%
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.neighbors import KernelDensity
from skada.da import ReweightDensity, GaussianReweightDensity, ClassifierReweightDensity
from skada.da import SubspaceAlignment, TransferComponentAnalysis
from skada.da import (
    OTmapping,
    EntropicOTmapping,
    ClassRegularizerOTmapping,
    LinearOTmapping,
    CORAL
)
from skada.datasets import make_shifted_datasets

# Use same random seed for multiple calls to make_datasets to
# ensure same distributions
RANDOM_SEED = 42

names = [
    "Without da",
    "Reweight Density",
    "Gaussian Reweight Density",
    "Classifier Reweight Density",
    "Subspace Alignment",
    "Transfer Component Analysis",
    "EMD Transport",
    "Sinkhorn Transport",
    "Sinkhorn Lpl1 Transport",
    "Linear Transport",
    "CORAL"
]

classifiers = [
    SVC(),
    ReweightDensity(
        base_estimator=SVC(),
        weight_estimator=KernelDensity(bandwidth=0.005),
    ),
    GaussianReweightDensity(base_estimator=SVC()),
    ClassifierReweightDensity(base_estimator=SVC()),
    SubspaceAlignment(base_estimator=SVC(), n_components=2),
    TransferComponentAnalysis(base_estimator=SVC(), n_components=2),
    OTmapping(base_estimator=SVC()),
    EntropicOTmapping(base_estimator=SVC()),
    ClassRegularizerOTmapping(base_estimator=SVC()),
    LinearOTmapping(base_estimator=SVC()),
    CORAL(base_estimator=SVC()),
]

datasets = [
    make_shifted_datasets(
        n_samples_source=20,
        n_samples_target=20,
        shift="covariate_shift",
        label="binary",
        noise=0.4,
        random_state=RANDOM_SEED
    ),
    make_shifted_datasets(
        n_samples_source=20,
        n_samples_target=20,
        shift="target_shift",
        label="binary",
        noise=0.4,
        random_state=RANDOM_SEED
    ),
    make_shifted_datasets(
        n_samples_source=20,
        n_samples_target=20,
        shift="concept_drift",
        label="binary",
        noise=0.4,
        random_state=RANDOM_SEED
    ),
    make_shifted_datasets(
        n_samples_source=20,
        n_samples_target=20,
        shift="subspace",
        label="binary",
        noise=0.4,
        random_state=RANDOM_SEED
    ),
]

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y, X_target, y_target = ds
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    ax = plt.subplot(len(datasets), len(classifiers) + 2, i)
    if ds_cnt == 0:
        ax.set_title("Source data")
    # Plot the source points
    ax.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap=cm_bright,
        alpha=0.5,
    )
    i += 1
    ax = plt.subplot(len(datasets), len(classifiers) + 2, i)
    if ds_cnt == 0:
        ax.set_title("Target data")
    # Plot the target points
    ax.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap=cm_bright,
        alpha=0.1,
    )
    ax.scatter(
        X_target[:, 0],
        X_target[:, 1],
        c=y_target,
        cmap=cm_bright,
        alpha=0.5,
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        if ds_cnt == 3 and name == "Reweight Density":
            continue
        ax = plt.subplot(len(datasets), len(classifiers) + 2, i)
        if name == "Without da":
            clf.fit(X, y)
        else:
            clf.fit(X, y, X_target)
        score = clf.score(X_target, y_target)
        DecisionBoundaryDisplay.from_estimator(
            clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5
        )

        # Plot the target points
        ax.scatter(
            X_target[:, 0],
            X_target[:, 1],
            c=y_target,
            cmap=cm_bright,
            alpha=0.5,
        )

        ax.set_xlim(x_min, x_max)

        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(
            x_max - 0.3,
            y_min + 0.3,
            ("%.2f" % score).lstrip("0"),
            size=15,
            horizontalalignment="right",
        )
        i += 1

plt.tight_layout()
plt.show()

# %%
