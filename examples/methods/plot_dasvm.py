"""
JDOT Regressor example
======================

This example shows how to use the JDOTRegressor [10] to learn a regression model
from source to target domain on a simple concept drift 2D exemple. We use a
simple Kernel Ridge Regression (KRR) as base estimator.

We compare the performance of the KRR on the source and target domain, and the
JDOTRegressor on the same task and illustrate the learned decision boundary and
the OT plan between samples estimated by JDOT.

[10] Courty, N., Flamary, R., Habrard, A., & Rakotomamonjy, A. (2017). Joint
 distribution optimal transportation for domain adaptation. Advances in neural
 information processing systems, 30.

"""

# Author: Remi Flamary
#
# License: BSD 3-Clause


# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from skada._self_labeling import DASVMEstimator
from skada.datasets import make_dataset_from_moons_distribution
from skada import source_target_split

from sklearn.base import clone

from sklearn.svm import SVC
