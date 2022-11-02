# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#
# License: MIT License

import numpy as np

from sklearn.decomposition import PCA

from .base import BaseSubspaceEstimator


class SubspaceAlignment(BaseSubspaceEstimator):
    """Estimator based on reweighting samples using density estimation.

    Parameters
    ----------
    base_estimator : estimator object
        The base estimator to fit on reweighted data.
    n_components : int, default=None
        The numbers of components to learn with PCA.
        Should be less or equal to the number of features
        of the data.

    References
    ----------
    .. [1]  Basura Fernando et. al. Unsupervised Visual
            Domain Adaptation Using Subspace Alignment.
            In IEEE International Conference on Computer Vision, 2013.
    """
    def __init__(
        self,
        base_estimator,
        n_components=None,
    ):
        super().__init__(base_estimator)

        self.n_components = n_components

    def predict_adapt(self, X, y, X_target, y_target=None):
        """Predict adaptation (weights, sample or labels)"""
        weights = None
        self.M_ = np.dot(self.PCA_source_.components_, self.PCA_target_.components_.T)
        X_ = np.dot(self.PCA_source_.transform(X), self.M_)
        return X_, y, weights

    def fit_adapt(self, X, y, X_target, y_target=None):
        """Fit adaptation parameters"""
        assert self.n_components <= X.shape[1], "n_components higher than n_features"

        self.PCA_source_ = PCA(self.n_components)
        self.PCA_target_ = PCA(self.n_components)
        self.PCA_source_.fit(X)
        self.PCA_target_.fit(X_target)

    def transform(self, X, domain='target'):

        assert domain in ['target', 'source']

        if domain == 'source':
            X_transform = np.dot(self.PCA_source_.transform(X), self.M_)
        else:
            X_transform = self.PCA_target_.transform(X)
        return X_transform
