# coding: utf-8
"""
=====================
SKADA logo generation
=====================


In this example we plot the logos of the SKADA toolbox.

This logo is that it is done 100% in Python and generated using
matplotlib and plotting the solution of the EMD solver from POT.
"""

# Author: Remi Flamary
#
# License: MIT License
# sphinx_gallery_thumbnail_number = 1
# %% Imports

import numpy as np
import pylab as pl

import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern"})


def get_lda_interp(xs, xt, alpha, len_=1):
    """Compute the LDA interpolation between two domains"""

    # compute means (clases assumed to be balanced)
    m1 = (1-alpha)*xs[:n//2, :].mean(0)+(alpha)*xt[:n//2, :].mean(0)
    m2 = (1-alpha)*xs[n//2:, :].mean(0)+(alpha)*xt[n//2:, :].mean(0)

    vo = m2-m1
    vo = vo / np.linalg.norm(vo)

    theta = np.pi/2
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    vor = rot.dot(vo)

    mm = 0.5*m1+0.5*m2

    return np.vstack((mm-len_*vor, mm+len_*vor))

# %% Generate Data


n = 40

np.random.seed(42)

xs = 0.2*np.random.randn(n, 2)

# class  specific change
xs[n//2:, 0] += 0.5
xs[n//2:, 1] -= 1

# classes 0 an 3 for blue/red colors
ys = np.zeros(n)
ys[n//2:] = 3

# global changes
xt = 0.15*np.random.randn(n, 2)
xt[:, 0] += 0.7
xt[:, 1] -= 0

# class specific change
xt[n//2:, 0] += 1
xt[n//2:, 1] -= 0.2

# class 7 for gray color (target without label)
yt = np.ones(n)*7


# %% Plot the small logo
nb = 10
alpha0 = 0.2

alphalist = np.linspace(0, 1, nb)

pl.figure(1, (2, 2))
pl.clf()
alpha = 0.7

# plot samples
pl.scatter(xs[ys == 0, 0], xs[ys == 0, 1], c='C0', cmap='tab10',
           vmax=10, alpha=alpha, label='Source data', marker='o')
pl.scatter(xs[ys == 3, 0], xs[ys == 3, 1], c='C3', cmap='tab10',
           vmax=10, alpha=alpha, label='Source data', marker='s')
pl.scatter(xt[ys == 0, 0], xt[ys == 0, 1], c='C7', cmap='tab10',
           vmax=10, alpha=alpha, label='Source data', marker='o')
pl.scatter(xt[ys == 3, 0], xt[ys == 3, 1], c='C7', cmap='tab10',
           vmax=10, alpha=alpha, label='Source data', marker='s')

# plot classifiers
ax = pl.axis()
for i in range(nb):
    xclass = get_lda_interp(xs, xt, alphalist[i], 2)
    pl.plot(xclass[:, 0], xclass[:, 1], color='C2', alpha=alpha0 +
            (1-alpha0)*alphalist[i], zorder=0, linewidth=2)
pl.axis(ax)
pl.axis('off')

# save to file
pl.savefig('skada_logo.pdf', bbox_inches='tight')
pl.savefig('skada_logo.svg', bbox_inches='tight', dpi=400)
pl.savefig('skada_logo.png', bbox_inches='tight', dpi=400)


# %%

# target data for full logo
xt2 = xt.copy()
xt2[:, 0] += 6.3


nb = 10
alpha0 = 0.2
alphalist = np.linspace(0, 1, nb)

pl.figure(2, (9, 1.5))
pl.clf()
alpha = 0.7

# plot samples
pl.scatter(xs[ys == 0, 0], xs[ys == 0, 1], c='C0', cmap='tab10',
           vmax=10, alpha=alpha, label='Source data', marker='o')
pl.scatter(xs[ys == 3, 0], xs[ys == 3, 1], c='C3', cmap='tab10',
           vmax=10, alpha=alpha, label='Source data', marker='s')
pl.scatter(xt2[ys == 0, 0], xt2[ys == 0, 1], c='C7', cmap='tab10',
           vmax=10, alpha=alpha, label='Source data', marker='o')
pl.scatter(xt2[ys == 3, 0], xt2[ys == 3, 1], c='C7', cmap='tab10',
           vmax=10, alpha=alpha, label='Source data', marker='s')

# plot classifiers
ax = pl.axis()
for i in range(nb):
    xclass = get_lda_interp(xs, xt2, alphalist[i], 2)
    pl.plot(xclass[:, 0], xclass[:, 1], color='C2', alpha=alpha0 +
            (1-alpha0)*alphalist[i], zorder=0, linewidth=2)
pl.text(1.3, -1.18, r'\bf\textsf{SKADA}', fontsize=80,
        usetex=True, zorder=0.5, color=(0.2, 0.2, 0.2))

pl.axis(ax)
pl.axis('off')

pl.savefig('skada_logo_full.pdf', bbox_inches='tight')
pl.savefig('skada_logo_full.svg', bbox_inches='tight', dpi=400)
pl.savefig('skada_logo_full.png', bbox_inches='tight', dpi=400)
