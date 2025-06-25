"""
Comparison of DA classification methods
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
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.neighbors import KernelDensity
from sklearn.svm import SVC

from skada import (
    CORAL,
    ClassRegularizerOTMapping,
    DensityReweight,
    DiscriminatorReweight,
    EntropicOTMapping,
    GaussianReweight,
    JDOTClassifier,
    KLIEPReweight,
    LinearOTMapping,
    MMDLSConSMapping,
    MMDTarSReweight,
    NearestNeighborReweight,
    OTLabelProp,
    OTMapping,
    SubspaceAlignment,
    TransferComponentAnalysis,
    TransferSubspaceLearning,
)
from skada.datasets import make_shifted_datasets

# Use same random seed for multiple calls to make_datasets to
# ensure same distributions
RANDOM_SEED = 42

names = [
    "Without da",
    "Reweight Density",
    "Gaussian Reweight",
    "Discr. Reweight",
    "KLIEPReweight",
    "1NN Reweight Density",
    "MMD TarS",
    "Subspace Alignment",
    "TCA",
    "TSL",
    "OT mapping",
    "Entropic OT mapping",
    "Class Reg. OT mapping",
    "Linear OT mapping",
    "OT Label Propagation",
    "CORAL",
    "JDOT",
    "MMD Loc-Scale mapping",
]

classifiers = [
    SVC(),
    DensityReweight(
        base_estimator=SVC().set_fit_request(sample_weight=True),
        weight_estimator=KernelDensity(bandwidth=0.5),
    ),
    GaussianReweight(SVC().set_fit_request(sample_weight=True)),
    DiscriminatorReweight(SVC().set_fit_request(sample_weight=True)),
    KLIEPReweight(SVC().set_fit_request(sample_weight=True), gamma=[1, 0.1, 0.001]),
    NearestNeighborReweight(SVC().set_fit_request(sample_weight=True)),
    MMDTarSReweight(SVC().set_fit_request(sample_weight=True), gamma=1),
    SubspaceAlignment(base_estimator=SVC(), n_components=1),
    TransferComponentAnalysis(base_estimator=SVC(), n_components=1, mu=0.5),
    TransferSubspaceLearning(base_estimator=SVC(), n_components=1),
    OTMapping(base_estimator=SVC()),
    EntropicOTMapping(base_estimator=SVC()),
    ClassRegularizerOTMapping(base_estimator=SVC()),
    LinearOTMapping(base_estimator=SVC()),
    OTLabelProp(base_estimator=SVC()),
    CORAL(base_estimator=SVC()),
    JDOTClassifier(base_estimator=SVC(), metric="hinge"),
    MMDLSConSMapping(base_estimator=SVC()),
]

datasets = [
    make_shifted_datasets(
        n_samples_source=20,
        n_samples_target=20,
        shift="covariate_shift",
        label="binary",
        noise=0.4,
        random_state=RANDOM_SEED,
        return_dataset=True,
    ),
    make_shifted_datasets(
        n_samples_source=20,
        n_samples_target=20,
        shift="target_shift",
        label="binary",
        noise=0.4,
        random_state=RANDOM_SEED,
        return_dataset=True,
    ),
    make_shifted_datasets(
        n_samples_source=20,
        n_samples_target=20,
        shift="conditional_shift",
        label="binary",
        noise=0.4,
        random_state=RANDOM_SEED,
        return_dataset=True,
    ),
    make_shifted_datasets(
        n_samples_source=20,
        n_samples_target=20,
        shift="subspace",
        label="binary",
        noise=0.4,
        random_state=RANDOM_SEED,
        return_dataset=True,
    ),
]

figure, axes = plt.subplots(len(classifiers) + 2, len(datasets), figsize=(9, 27))
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y, sample_domain = ds.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )
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

    ax = axes[1, ds_cnt]

    if ds_cnt == 0:
        ax.set_ylabel("Target data")
    # Plot the target points
    ax.scatter(
        Xs[:, 0],
        Xs[:, 1],
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
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        if name == "Without da":
            clf.fit(Xs, ys)
        else:
            clf.fit(X, y, sample_domain=sample_domain)
        score = clf.score(Xt, yt)
        DecisionBoundaryDisplay.from_estimator(
            clf,
            X,
            cmap=cm,
            alpha=0.8,
            ax=ax,
            eps=0.5,
            response_method="predict",
        )

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
        i += 1

plt.tight_layout()
plt.show()
