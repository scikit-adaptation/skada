"""
    dasvm
==========================================

This example illustrates the dsvm method from [1].

.. [1]  Domain Adaptation Problems: A DASVM Classification
        Technique and a Circular Validation Strategy
        Lorenzo Bruzzone, Fellow, IEEE, and Mattia Marconcini, Member, IEEE

"""

# Author: Ruben Bueno <ruben.bueno@polytechnique.edu>
# dasvm implementation

import numpy as np
import matplotlib.pyplot as plt

from skada.datasets import make_shifted_datasets
from skada._dasvm import BaseDasvmAdapter
from skada.utils import check_X_y_domain, source_target_split
from skada._pipeline import make_da_pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


# base_estimator can be any classifier equipped with decision function:
# LogisticRegression(random_state=0), SVC(gamma='auto'), etc...
base_estimator = LogisticRegression(random_state=0)

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

axis[1].scatter(Xt[:, 0], Xt[:, 1], c=yt)
axis[1].set_xlim(xlim)
axis[1].set_ylim(ylim)

E = make_da_pipeline(
    StandardScaler(), BaseDasvmAdapter(k=5)).fit(X, y, sample_domain=sample_domain)


figure, axis = plt.subplots(1, 2)
a = []
for i in [0, -1]:
    e = (SVC(gamma='auto').fit(Xs, ys) if i == 0 else E)
    x_points = np.linspace(xlim[0], xlim[1], 200)
    y_points = np.linspace(ylim[0], ylim[1], 200)
    # Plotting a red hyperplane
    X = np.array([[x, y] for x in x_points for y in y_points])
    a.append(axis[i].scatter(X[:, 0], X[:, 1], c=e.decision_function(X)))
    axis[i].set_xlim(xlim)
    axis[i].set_ylim(ylim)
figure.colorbar(a[-1])
plt.show()

