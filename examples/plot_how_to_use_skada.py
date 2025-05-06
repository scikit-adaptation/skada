"""
How to use SKADA
====================================================

This is a short beginners guide to get started with SKADA and perform domain adaptation
on a simple dataset. It illustrates the API choice specific to DA.
For better readability, only the use of SKADA is provided and the plotting code
with matplotlib is hidden (it is still available in the source file of the example).
"""

# Author: Remi Flamary
#         Maxence Barnèche
#
# License: BSD 3-Clause
# sphinx_gallery_thumbnail_number = 1

# necessary imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from skada import (
    CORAL,
    CORALAdapter,
    GaussianReweightAdapter,
    PerDomain,
    SelectSource,
    SelectSourceTarget,
    make_da_pipeline,
    source_target_split,
)
from skada.datasets import make_shifted_datasets
from skada.metrics import PredictionEntropyScorer
from skada.model_selection import SourceTargetShuffleSplit

# %%
# DA dataset
# ----------
#
# We generate a simple 2D DA dataset. Note that DA datasets provided by SKADA
# are organized as follows:
#
# * :code:`X` is the input data, including the source and the target samples
# * :code:`y` is the output data to be predicted (labels on target samples are not
#   used when fitting the DA estimator)
# * :code:`sample_domain` encodes the domain of each sample (integer >=0 for
#   source and <0 for target)
#
# Four different types of dataset shifts are handled
#
# Covariate shift
# ~~~~~~~~~~~~~~~
#
# Covariate shift is characterised by a change of distribution in one or more of
# the independent variables from the input data.

# create a DA dataset with covariate shift
X, y, sample_domain = make_shifted_datasets(
    20, 20, shift="covariate_shift", random_state=42
)


# sphinx_gallery_start_ignore
def decision_borders_plot(X, model):
    # Create a meshgrid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, num=100), np.linspace(y_min, y_max, num=100)
    )

    # Predict on every point of the meshgrid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision borders
    plt.contourf(xx, yy, Z, alpha=0.4, cmap="tab10", vmax=9)


def source_target_comparison(X, y, sample_domain, title, prediction=False, model=None):
    Xs, Xt, ys, yt = source_target_split(X, y, sample_domain=sample_domain)
    plt.subplot(1, 2, 1)
    plt.scatter(
        Xs[:, 0], Xs[:, 1], c=ys, cmap="tab10", vmax=9, label="Source", alpha=0.8
    )
    plt.xticks([])
    plt.yticks([])
    plt.title("Source data")
    if prediction:
        decision_borders_plot(X, model)
    ax = plt.axis()

    plt.subplot(1, 2, 2)
    plt.scatter(
        Xt[:, 0], Xt[:, 1], c=yt, cmap="tab10", vmax=9, label="Target", alpha=0.8
    )
    plt.xticks([])
    plt.yticks([])
    plt.title("Target data")
    if prediction:
        decision_borders_plot(X, model)
    plt.axis(ax)

    plt.suptitle(title)


plt.figure(1, (7.5, 3.625))
source_target_comparison(X, y, sample_domain, "Covariate shift")
plt.tight_layout()
# sphinx_gallery_end_ignore

# %%
# Target shift
# ~~~~~~~~~~~~
#
# Target shift (or prior probability shift) is characterised by a change in
# target variable distribution while the source data distribution remains the same.

# create a DA dataset with target shift
X, y, sample_domain = make_shifted_datasets(
    20, 20, shift="target_shift", random_state=42
)

# sphinx_gallery_start_ignore
plt.figure(2, (7.5, 3.625))
source_target_comparison(X, y, sample_domain, "Target shift")
plt.tight_layout()
# sphinx_gallery_end_ignore

# %%
# Conditional shift
# ~~~~~~~~~~~~~~~~~
#
# Conditional shift (or concept drift) is characterised by a change in
# the relation between input and output variables.

# create a DA dataset with conditional shift
X, y, sample_domain = make_shifted_datasets(
    20, 20, shift="conditional_shift", random_state=42
)

# sphinx_gallery_start_ignore
plt.figure(3, (7.5, 3.625))
source_target_comparison(X, y, sample_domain, "Conditional shift")
plt.tight_layout()
# sphinx_gallery_end_ignore

# %%
# Subspace shift
# ~~~~~~~~~~~~~~
#
# Subspace shift is characterised by a change in data distribution where
# there exists a subspace such that the projection of the data
# on that subspace keeps the same distribution

# create a DA dataset with subspace
X, y, sample_domain = make_shifted_datasets(20, 20, shift="subspace", random_state=42)

# sphinx_gallery_start_ignore
plt.figure(4, (7.5, 3.625))
source_target_comparison(X, y, sample_domain, "Subspace shift")
plt.tight_layout()
# sphinx_gallery_end_ignore

# %%
# DA Classifier estimator
# -----------------------
#
# SKADA estimators are used like scikit-learn estimators. The only difference is
# that the :code:`sample_domain` array must be passed by name when fitting the
# estimator.

# split source and target for visualization and source-target comparison
Xs, Xt, ys, yt = source_target_split(X, y, sample_domain=sample_domain)
sample_domain_s = np.ones(Xs.shape[0])
sample_domain_t = -np.ones(Xt.shape[0]) * 2

# create a DA estimator
clf = CORAL()

# train on all data
clf.fit(X, y, sample_domain=sample_domain)

# estimator is designed to predict on target by default
yt_pred = clf.predict(Xt)

# accuracy on source and on target
acc_s = clf.score(Xs, ys)
acc_t = clf.score(Xt, yt)

# sphinx_gallery_start_ignore
plt.figure(5, (7.5, 3.625))
source_target_comparison(
    X, y, sample_domain, "Predictions on conditional shift", prediction=True, model=clf
)
plt.tight_layout()
# accuracy on source and target
print("Accuracy on source:", acc_s)
print("Accuracy on target:", acc_t)
# sphinx_gallery_end_ignore

# %%
# DA estimator in a pipeline
# -----------------------------
#
# SKADA estimators can be used as the final estimator of a scikit-learn pipeline.
# Again, the only difference is that the :code:`sample_domain` array must be passed
# by name during in fit.


# create a DA pipeline
pipe = make_pipeline(StandardScaler(), CORAL(base_estimator=SVC()))
pipe.fit(X, y, sample_domain=sample_domain)

print("Accuracy on target:", pipe.score(Xt, yt))

# %%
# DA Adapter pipeline
# -------------------
#
# Several SKADA estimators include a data adapter that transforms the input data
# so that a scikit-learn estimator can be used. For those methods, SKADA
# provides a :code:`Adapter` class that can be used in a DA pipeline from
# :code:`make_da_pipeline`.
#
# Here is an example with the CORAL and GaussianReweight adapters.
#
# .. WARNING::
#
#   Note that as illustrated below for reweighting adapters, one needs a
#   subsequent estimator that takes :code:`sample_weight` as an input parameter.
#   This can be done using the :code:`set_fit_request` method of the estimator
#   by calling :code:`.set_fit_request(sample_weight=True)`.
#   If the estimator (for pipeline or DA estimator) does not
#   require sample weights, the DA pipeline will raise an error.


# create a DA pipeline with CORAL adapter
pipe = make_da_pipeline(StandardScaler(), CORALAdapter(), SVC())
pipe.fit(X, y, sample_domain=sample_domain)

print("Accuracy on target:", pipe.score(Xt, yt))

# create a DA pipeline with GaussianReweight adapter
# (does not work well on conditional shift).
pipe = make_da_pipeline(
    StandardScaler(),
    GaussianReweightAdapter(),
    LogisticRegression().set_fit_request(sample_weight=True),
)
pipe.fit(X, y, sample_domain=sample_domain)

print("Accuracy on target:", pipe.score(Xt, yt))

# %%
# DA estimators with score cross-validation
# -------------------------------------------
#
# DA estimators are compatible with scikit-learn cross-validation functions.
# Note that the :code:`sample_domain` array must be passed in the :code:`params`
# dictionary of the :code:`cross_val_score` function.


# splitter for cross-validation of score
cv = SourceTargetShuffleSplit(random_state=0)

# DA scorer not using target labels (not available in DA)
scorer = PredictionEntropyScorer()

clf = CORAL(SVC(probability=True))  # needs probability for entropy score

# cross-validation
scores = cross_val_score(
    clf, X, y, params={"sample_domain": sample_domain}, cv=cv, scoring=scorer
)

print(f"Entropy score: {scores.mean():1.2f} (+-{scores.std():1.2f})")

# %%
# DA estimator with grid search
# -----------------------------
#
# DA estimators are also compatible with scikit-learn grid search functions.
# Note that the :code:`sample_domain` array must be passed in the :code:`fit`
# method of the grid search.


reg_coral = [0.1, 0.5, 1, "auto"]

clf = make_da_pipeline(StandardScaler(), CORALAdapter(), SVC(probability=True))

# grid search
grid_search = GridSearchCV(
    estimator=clf,
    param_grid={"coraladapter__reg": reg_coral},
    cv=SourceTargetShuffleSplit(random_state=0),
    scoring=PredictionEntropyScorer(),
)

grid_search.fit(X, y, sample_domain=sample_domain)

print("Best regularization parameter:", grid_search.best_params_["coraladapter__reg"])
print("Accuracy on target:", np.mean(grid_search.predict(Xt) == yt))


# %%
# Advanced DA pipeline
# --------------------
#
# The DA pipeline can be used with any estimator and any adapter. But more
# importantly all estimators in the pipeline are automatically wrapped in what
# we call in skada a `Selector`. The selector is a wrapper that allows you to
# choose which data is passed during fit and predict/transform.
#
# In the following example, one StandardScaler is trained per domain. Then
# a single SVC is trained on source data only. When predicting on target data the
# pipeline will automatically use the StandardScaler trained on target and the SVC
# trained on source.

# create a DA pipeline with SelectSourceTarget estimators

pipe = make_da_pipeline(
    SelectSourceTarget(StandardScaler()),
    SelectSource(SVC()),
)

pipe.fit(X, y, sample_domain=sample_domain)

print("Accuracy on source:", pipe.score(Xs, ys, sample_domain=sample_domain_s))
print("Accuracy on target:", pipe.score(Xt, yt))  # target by default


# %%
# Similarly one can use the PerDomain selector to train a different estimator
# per domain. This allows to handle multiple source and target domains. In this
# case :code:`sample_domain` must be provided to fit and predict/transform.

pipe = make_da_pipeline(
    PerDomain(StandardScaler()),
    SelectSource(SVC()),
)

pipe.fit(X, y, sample_domain=sample_domain)

print("Accuracy on all data:", pipe.score(X, y, sample_domain=sample_domain))

# %%
# One can use a default selector on the whole pipeline which allows for
# instance to train the whole pipeline only on the source data as follows:

pipe_train_on_source = make_da_pipeline(
    StandardScaler(),
    SVC(),
    default_selector=SelectSource,
)

pipe_train_on_source.fit(X, y, sample_domain=sample_domain)
print("Accuracy on source:", pipe_train_on_source.score(Xs, ys))
print("Accuracy on target:", pipe_train_on_source.score(Xt, yt))

# %%
# One can also use a default selector on the whole pipeline but overwrite it for
# the last estimator. In the example below a :code:`StandardScaler` and a
# :code:`PCA` are estimated per domain but the final SVC is trained on source data only.

pipe_perdomain = make_da_pipeline(
    StandardScaler(),
    PCA(n_components=2),
    SelectSource(SVC()),
    default_selector=SelectSourceTarget,
)

pipe_perdomain.fit(X, y, sample_domain=sample_domain)
print(
    "Accuracy on source:", pipe_perdomain.score(Xs, ys, sample_domain=sample_domain_s)
)
print(
    "Accuracy on target:", pipe_perdomain.score(Xt, yt, sample_domain=sample_domain_t)
)
