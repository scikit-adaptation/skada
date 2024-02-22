"""
How to use SKADA
====================================================

This is a short example of how to use SKADA to perform domain adaptation
on a simple dataset will illustration of the API choice specific to DA.
"""

# Author: Remi Flamary
#
# License: MIT License
# sphinx_gallery_thumbnail_number = 1

# %% imports
import numpy as np
from sklearn.model_selection import GridSearchCV
from skada.metrics import PredictionEntropyScorer
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from skada import CORALAdapter, GaussianReweightDensityAdapter
from skada import make_da_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from skada import CORAL
import matplotlib.pyplot as plt

from skada import source_target_split
from skada.datasets import make_shifted_datasets

# %%
# DA dataset
# ----------
#
# We generate a simple 2D DA dataset. Note that DA data
# are organized with all source and target sample as follows:
#
# * :code:`X` is the input data
# * :code:`y` is the output data to be predicted (labels on target samples are not
#   used)
# * :code:`sample_domain` is the domain of each sample (integer <=0 for
#   source and >0 for target)

# Get DA dataset
X, y, sample_domain = make_shifted_datasets(
    20, 20, shift="concept_drift", random_state=42)

# split source and target for visualization
Xs, Xt, ys, yt = source_target_split(X, y, sample_domain=sample_domain)

# plot data
plt.figure(1, (10, 5))

plt.subplot(1, 2, 1)
plt.scatter(Xs[:, 0], Xs[:, 1], c=ys, cmap='tab10', vmax=9, label="Source")
plt.title("Source data")
ax = plt.axis()

plt.subplot(1, 2, 2)
plt.scatter(Xt[:, 0], Xt[:, 1], c=yt, cmap='tab10', vmax=9, label="Target")
plt.axis(ax)
plt.title("Target data")

# %%
# DA Classifier estimator
# -----------------------
#
# DA estimators are used exactly as regular estimators. The only difference is
# that the :code:`sample_domain` array must be passed (by name).


# create a DA estimator
clf = CORAL()

# train on all data
clf.fit(X, y, sample_domain=sample_domain)

# estimator is designed to predict on target by default
yt_pred = clf.predict(Xt)

# accuracy on source and target
print('Accuracy on source:', clf.score(Xs, ys))
print('Accuracy on target:', clf.score(Xt, yt))

# %% DA estimator in a pipeline
# -----------------------------
#
# DA estimators can be used as final estimator in the sklearn pipeline. The only
# difference is that the :code:`sample_domain` array must be passed (by name)
# during the fit.


# create a DA pipeline
pipe = make_pipeline(StandardScaler(), CORAL(base_estimator=SVC()))
pipe.fit(X, y, sample_domain=sample_domain)

print('Accuracy on target:', pipe.score(Xt, yt))

# %%
# DA Adapter pipeline
# -------------------
#
# Many DA method correspond to a data adapter that can be used to transform
# data such that a regular estimator can be used. For those methods, skada
# provides a :code:`Adapter` class that can be used in a DA pipeline from
# :code: .
#
# Here is an example with the CORAL and GaussianReweight adapter.
#
# .. WARNING::
#   Note that as illustrated below for
#   adapters that reweight the data, one needs a subsequent estimator that
#   requires the weighs as input. This can be done with :code:`set_fit_request`
#   method of the estimator by executing
#   :code:`.set_fit_request(sample_weight=True)`). If the estimator (for
#   pipeline or DA estimator) does not
#   require sample weights, the DA pipeline will raise an error.


# create a DA pipeline with CORAL adapter
pipe = make_da_pipeline(StandardScaler(), CORALAdapter(), SVC())
pipe.fit(X, y, sample_domain=sample_domain)

print('Accuracy on target:', pipe.score(Xt, yt))

# create a DA pipeline with GaussianReweight adapter
pipe = make_da_pipeline(StandardScaler(),
                        GaussianReweightDensityAdapter(),
                        LogisticRegression().set_fit_request(sample_weight=True))
pipe.fit(X, y, sample_domain=sample_domain)

print('Accuracy on target:', pipe.score(Xt, yt))

# %%
# DA estimator with cross-validation of score
# -------------------------------------------
#
# DA estimators can be used with cross-validation as regular estimators using
# functions from sklearn. Note that the :code:`sample_domain` array must be
# passed in the :code:`params` dictionary of the :code:`cross_val_score` function.


# splitter for cross-validation of score
cv = ShuffleSplit(random_state=0)

# DA scorer not using target labels (not available in DA)
scorer = PredictionEntropyScorer()

clf = CORAL(SVC(probability=True))  # needs probability for entropy score

# cross-validation
scores = cross_val_score(clf, X, y, params={'sample_domain': sample_domain},
                         cv=cv, scoring=scorer)

print('Entropy score: {:1.2f} (+-{:1.2f})'.format(scores.mean(), scores.std()))

# %%
# DA estimator with grid search
# -----------------------------
#
# DA estimators can be used with grid search as regular estimators using
# functions from sklearn. Note that the :code:`sample_domain` array must be
# passed in the :code:`fit` method of the grid search.


reg_coral = [0.1, 0.5, 1, "auto"]

clf = make_da_pipeline(StandardScaler(), CORALAdapter(), SVC(probability=True))

# grid search
grid_search = GridSearchCV(
    estimator=clf,
    param_grid={"coraladapter__reg": reg_coral},
    cv=ShuffleSplit(random_state=0),
    scoring=PredictionEntropyScorer(),
)


grid_search.fit(X, y, sample_domain=sample_domain)

print('Best regularization parameter:', grid_search.best_params_['coraladapter__reg'])
print('Accuracy on target:', np.mean(grid_search.predict(Xt) == yt))
