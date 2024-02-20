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
import math

from skada._dasvm import DASVMEstimator
from sklearn.base import clone

from sklearn.svm import SVC


# base_estimator can be any classifier equipped with `decision_function` such as:
# SVC(gamma='auto'), LogisticRegression(random_state=0), etc...
base_estimator = SVC(kernel="Linear")

xlim = (-2.2, 2.5)
ylim = (-1.5, 2)

figure, axis = plt.subplots(1, 2)

N = 100
theta_s = np.concatenate((
    np.linspace(10, 170, N),
    np.linspace(190, 350, N),
    ))*math.pi/180

theta_target = 20
theta_t = (np.concatenate((
    np.linspace(10, 170, N),
    np.linspace(190, 350, N),
    ))+theta_target)*math.pi/180

Xs = np.array([
    (1+np.random.normal(0, 0.1, theta_s.shape[0]))*np.cos(theta_s),
    (1+np.random.normal(0, 0.1, theta_s.shape[0]))*np.sin(theta_s)
    ]).T
Xt = np.array([
    (1+np.random.normal(0, 0.1, theta_t.shape[0]))*np.cos(theta_t),
    (1+np.random.normal(0, 0.1, theta_t.shape[0]))*np.sin(theta_t)
    ]).T
X = np.concatenate((Xs, Xt))

ys = np.array([0]*N+[1]*N)
yt = np.copy(ys)
y = np.concatenate((ys, yt))

sample_domain = np.array([1]*2*N+[-2]*2*N)

axis[0].scatter(Xs[:, 0], Xs[:, 1], c=ys)
axis[0].set_xlim(xlim)
axis[0].set_ylim(ylim)
axis[0].set_title("source data points")

axis[1].scatter(Xt[:, 0], Xt[:, 1], c=yt)
axis[1].set_xlim(xlim)
axis[1].set_ylim(ylim)
axis[1].set_title("target data points")

figure.suptitle("data points", fontsize=20)

estimator = DASVMEstimator(
    base_estimator=clone(base_estimator), k=10,
    save_estimators=True, save_indices=True).fit(
    X, y, sample_domain=sample_domain)

epsilon = 0.05
N = 4
K = len(estimator.estimators)//N
figure, axis = plt.subplots(1, N+1)
for i in list(range(0, N*K, K)) + [-1]:
    j = i//K if i != -1 else -1
    e = estimator.estimators[i]
    x_points = np.linspace(xlim[0], xlim[1], 200)
    y_points = np.linspace(ylim[0], ylim[1], 200)
    X = np.array([[x, y] for x in x_points for y in y_points])
    axis[j].scatter(
        X[:, 0], X[:, 1], c=e.decision_function(X), alpha=0.02)

    # plot margins
    X_ = X[np.absolute(e.decision_function(X)-1) < epsilon]
    axis[j].scatter(
        X_[:, 0], X_[:, 1], c=[1]*X_.shape[0],
        alpha=1, cmap="gray", s=[0.1]*X_.shape[0])
    X_ = X[np.absolute(e.decision_function(X)+1) < epsilon]
    axis[j].scatter(
        X_[:, 0], X_[:, 1], c=[1]*X_.shape[0],
        alpha=1, cmap="gray", s=[0.1]*X_.shape[0])
    X_ = X[np.absolute(e.decision_function(X)) < epsilon]
    axis[j].scatter(
        X_[:, 0], X_[:, 1], c=[1]*X_.shape[0],
        alpha=1, cmap="autumn", s=[0.1]*X_.shape[0])

    X = np.concatenate((
        Xs[~estimator.indices_source_deleted[i]],
        Xt[estimator.indices_target_added[i]]))

    if sum(estimator.indices_target_added[i]) > 0:
        semi_labels = e.predict(Xt[estimator.indices_target_added[i]])
    else:
        semi_labels = np.array([])
    a = axis[j].scatter(X[:, 0], X[:, 1], c=np.concatenate((
        ys[~estimator.indices_source_deleted[i]],
        semi_labels)))
    X = Xt[~estimator.indices_target_added[i]]
    axis[j].scatter(
        X[:, 0], X[:, 1], cmap="gray",
        c=[0.5]*X.shape[0], alpha=0.5, vmax=1, vmin=0)

    axis[j].set_xlim(xlim)
    axis[j].set_ylim(ylim)
figure.colorbar(a)
figure.suptitle("reasulting predictions", fontsize=20)
plt.show()
