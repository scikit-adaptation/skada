"""
    plot for the dasvm estimator
==========================================

This example illustrates the dsvm method from [1].

.. [1]  Domain Adaptation Problems: A DASVM Classification
        Technique and a Circular Validation Strategy
        Lorenzo Bruzzone, Fellow, IEEE, and Mattia Marconcini, Member, IEEE

"""

# Author: Ruben Bueno <ruben.bueno@polytechnique.edu>
# dasvm plots

import numpy as np
import matplotlib.pyplot as plt

from skada.datasets import make_shifted_datasets
from skada._dasvm import DASVMEstimator
from skada.utils import check_X_y_domain, source_target_split
from sklearn.base import clone

from sklearn.svm import SVC


# base_estimator can be any classifier equipped with `decision_function` such as:
# SVC(gamma='auto'), LogisticRegression(random_state=0), etc...
base_estimator = SVC(gamma='auto')

xlim = (-2.2, 4.2)
ylim = (-2, 4.2)

figure, axis = plt.subplots(1, 2)

X, y, sample_domain = make_shifted_datasets(
    n_samples_source=20,
    n_samples_target=15,
    shift="covariate_shift",
    noise=None,
    label="binary",
)
X, y, sample_domain = check_X_y_domain(X, y, sample_domain)
Xs, Xt, ys, yt = source_target_split(
    X, y, sample_domain=sample_domain
)


axis[0].scatter(Xs[:, 0], Xs[:, 1], c=ys)
axis[0].set_xlim(xlim)
axis[0].set_ylim(ylim)
axis[0].set_title("source data points")

axis[1].scatter(Xt[:, 0], Xt[:, 1], c=yt)
axis[1].set_xlim(xlim)
axis[1].set_ylim(ylim)
axis[1].set_title("target data points")

figure.suptitle("data points", fontsize=20)

E = DASVMEstimator(
    base_estimator=clone(base_estimator), k=3).fit(
    X, y, sample_domain=sample_domain)


figure, axis = plt.subplots(1, 2)
for i in [0, -1]:
    e = (base_estimator.fit(Xs, ys) if i == 0 else E)
    x_points = np.linspace(xlim[0], xlim[1], 200)
    y_points = np.linspace(ylim[0], ylim[1], 200)
    X = np.array([[x, y] for x in x_points for y in y_points])
    a = axis[i].scatter(X[:, 0], X[:, 1], c=e.decision_function(X))
    axis[i].set_xlim(xlim)
    axis[i].set_ylim(ylim)
    axis[i].set_title((
        "for SVC estimator on source data" if i == 0
        else "for DASVMEstimator on target data"))
figure.colorbar(a)
figure.suptitle("reasulting decision function", fontsize=20)
plt.show()
