"""
    Plot for the dasvm estimator
==========================================

This example illustrates the dsvm method from [1].

.. [1]  Domain Adaptation Problems: A DASVM Classification
        Technique and a Circular Validation Strategy
        Lorenzo Bruzzone, Fellow, IEEE, and Mattia Marconcini, Member, IEEE

"""

# Author: Ruben Bueno
#
# License: BSD 3-Clause
# sphinx_gallery_thumbnail_number = 1

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from skada._self_labeling import DASVMEstimator
from skada.datasets import make_dataset_from_moons_distribution
from skada import source_target_split

from sklearn.base import clone

from sklearn.svm import SVC


RANDOM_SEED = 42

# base_estimator can be any classifier equipped with `decision_function` such as:
# SVC(gamma='auto'), LogisticRegression(random_state=0), etc...
base_estimator = SVC()

target_marker = "s"
source_marker = "o"
xlim = (-1.5, 2.4)
ylim = (-1, 1.3)

figure, axis = plt.subplots(1, 2)

"""
    We generate our dataset
------------------------------------------
"""

X, y, sample_domain = make_dataset_from_moons_distribution(
    pos_source=[0.1, 0.2, 0.3, 0.4],
    pos_target=[0.6, 0.7, 0.8, 0.9],
    n_samples_source=10,
    n_samples_target=10,
    noise=0.1,
    random_state=RANDOM_SEED
)

Xs, Xt, ys, yt = source_target_split(
    X, y, sample_domain=sample_domain
)

"""
    Plots of the dataset
------------------------------------------
As we can see, the source and target datasets have different
distributions of the points but have the same labels for
the same x-values.
We are then in the case of covariate shift
"""

axis[0].scatter(Xs[:, 0], Xs[:, 1], c=ys, marker=source_marker)
axis[0].set_xlim(xlim)
axis[0].set_ylim(ylim)
axis[0].set_title("source data points")

axis[1].scatter(Xt[:, 0], Xt[:, 1], c=yt, marker=target_marker)
axis[1].set_xlim(xlim)
axis[1].set_ylim(ylim)
axis[1].set_title("target data points")

figure.suptitle("data points", fontsize=20)

plt.show()
