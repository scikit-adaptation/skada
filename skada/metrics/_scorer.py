# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD 3-Clause
import numpy as np

from sklearn.neighbors import KernelDensity
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, check_scoring
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sklearn.preprocessing import Normalizer
from sklearn.utils.extmath import softmax


class SupervisedScorer:
    """Compute score on supervised dataset.

    Parameters
    ----------
    X_test : array-like
        The test data used to compute the score.
    y_test : array-like
        The test label used to compute the score.
    scoring : str or callable, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        If None, the provided estimator object's `score` method is used.
    greater_is_better : bool, default=True
        Whether `scorer` is a score function (default), meaning high is
        good, or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the `scorer`.

    """

    def __init__(self, X_test, y_test, scoring=None, greater_is_better=True):
        self.X_test = X_test
        self.y_test = y_test
        self.scoring = scoring
        self._sign = 1 if greater_is_better else -1

    def __call__(self, estimator, X, y):
        scorer = check_scoring(estimator, self.scoring)
        return self._sign * scorer(estimator, self.X_test, self.y_test)


class ImportanceWeightedScorer:
    """Score based on source data using sample weight.

    See [1]_ for details.

    Parameters
    ----------
    X_target : array-like
        The target data used by the scorer to compute the sample weights.
    weight_estimator : estimator object, optional
        The estimator to use to estimate the densities of source and target
        observations. If None, a KernelDensity estimator is with
        default parameters used.
    scoring : str or callable, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        If None, the provided estimator object's `score` method is used.
    greater_is_better : bool, default=True
        Whether `scorer` is a score function (default), meaning high is
        good, or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the `scorer`.


    Attributes
    ----------
    weight_estimator_source_ : object
        The estimator object fitted on the source data.
    weight_estimator_target_ : object
        The estimator object fitted on the target data.

    References
    ----------
    .. [1]  Masashi Sugiyama. Covariate Shift Adaptation
            by Importance Weighted Cross Validation.
            Journal of Machine Learning Research, 2007.
    """

    def __init__(
        self,
        X_target,
        weight_estimator=None,
        scoring=None,
        greater_is_better=True,
    ):
        self.X_target = X_target
        self.weight_estimator = weight_estimator
        self.scoring = scoring
        self._sign = 1 if greater_is_better else -1

    def _fit_adapt(self, X, X_target):
        """Fit adaptation parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The source data.
        X_target : array-like, shape (n_samples, n_features)
            The target data.

        Returns
        -------
        self : object
            Returns self.
        """
        weight_estimator = self.weight_estimator
        if weight_estimator is None:
            weight_estimator = KernelDensity()
        self.weight_estimator_source_ = clone(weight_estimator)
        self.weight_estimator_target_ = clone(weight_estimator)
        self.weight_estimator_source_.fit(X)
        self.weight_estimator_target_.fit(X_target)

    def __call__(self, estimator, X, y):
        scorer = check_scoring(estimator, self.scoring)
        self._fit_adapt(X, self.X_target)
        ws = self.weight_estimator_source_.score_samples(X)
        wt = self.weight_estimator_target_.score_samples(X)
        weights = np.exp(wt - ws)
        weights /= weights.sum()
        return self._sign * scorer(
            estimator,
            X,
            y,
            sample_weight=weights,
        )


class PredictionEntropyScorer:
    """Score based on the entropy of predictions on unsupervised dataset.

    See [1]_ for details.

    Parameters
    ----------
    X_test : array-like
        The test data used by the scorer to compute the entropy.
    greater_is_better : bool, default=False
        Whether `scorer` is a score function (default), meaning high is
        good, or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the `scorer`.

    References
    ----------
    .. [1]  Pietro Morerio. Minimal-Entropy correlation alignment
            for unsupervised deep domain adapation.
            ICLR, 2018.
    """

    def __init__(self, X_test, greater_is_better=False):
        self.X_test = X_test
        self._sign = 1 if greater_is_better else -1

    def __call__(self, estimator, X, y):
        if not hasattr(estimator, "predict_proba"):
            raise AttributeError(
                "The estimator passed should "
                "have a 'predict_proba' method. "
                "The estimator %r does not." % estimator
            )
        proba = estimator.predict_proba(self.X_test)
        if hasattr(estimator, "predict_log_proba"):
            log_proba = estimator.predict_log_proba(self.X_test)
        else:
            log_proba = np.log(proba + 1e-7)
        entropy = np.sum(-proba * log_proba, axis=1)
        return - np.mean(entropy)


class SoftNeighborhoodDensity:
    """Score based on the entropy of similarity between unsupervised dataset.

    See [1]_ for details.

    Parameters
    ----------
    X_test : array-like
        The test data used by the scorer to similarity matrix.
    T :  float
        Temperature in the Eq. 2 in [1]_.
        Default is set to 0.05, the value proposed in the paper.
    greater_is_better : bool, default=True
        Whether `scorer` is a score function (default), meaning high is
        good, or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the `scorer`.

    References
    ----------
    .. [1]  Kuniaki Saito. Tune it the Right Way: Unsupervised Validation
            of Domain Adaptation via Soft Neighborhood Density.
            International Conference on Computer Vision, 2021.
    """

    def __init__(self, X_test, T=0.05, greater_is_better=True):
        self.X_test = X_test
        self.T = T
        self._sign = 1 if greater_is_better else -1

    def __call__(self, estimator, X, y):
        if not hasattr(estimator, "predict_proba"):
            raise AttributeError(
                "The estimator passed should"
                "have a 'predict_proba' method."
                "The estimator %r does not." % estimator
            )
        proba = estimator.predict_proba(self.X_test)
        proba = Normalizer(norm="l2").fit_transform(proba)

        similarity_matrix = proba @ proba.T / self.T
        np.fill_diagonal(similarity_matrix, - np.diag(similarity_matrix))
        similarity_matrix = softmax(similarity_matrix)

        entropy = np.sum(- similarity_matrix * np.log(similarity_matrix), axis=1)
        return self._sign * np.mean(entropy)


class DeepEmbeddedValidation:
    """Loss based on source data using features represention to weight samples.

    See [1]_ for details.

    Parameters
    ----------
    X_test : array-like
        The target data used by the scorer to compute the score.
    domain_classifier : sklearn classifier, optional
        Classifier used to predict the domains. If None, a
        LogisticRegression is used.
    loss_func : callable
        Loss function with signature
        `loss_func(y, y_pred, **kwargs)`.
        The loss function need not to be reduced.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for train_test_split. Pass an int
        for reproducible output across multiple function calls.
    greater_is_better : bool, default=False
        Whether `scorer` is a score function (default), meaning high is
        good, or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the `scorer`.

    References
    ----------
    .. [1]  Kaichao You. Towards Accurate Model Selection
            in Deep Unsupervised Domain Adaptation.
            ICML, 2019.
    """

    def __init__(
        self,
        X_test,
        domain_classifier=None,
        loss_func=None,
        random_state=None,
        greater_is_better=False,
    ):
        self.X_test = X_test
        self.domain_classifier = domain_classifier
        self._loss_func = (
            loss_func if loss_func is not None else self._no_reduc_log_loss
        )
        self.random_state = random_state
        self._sign = 1 if greater_is_better else -1

    def _no_reduc_log_loss(self, y, y_pred):
        return np.array(
            [
                log_loss(y[i : i + 1], y_pred[i : i + 1], labels=np.unique(y))
                for i in range(len(y))
            ]
        )

    def _fit_adapt(self, features, features_target):
        domain_classifier = self.domain_classifier
        if domain_classifier is None:
            domain_classifier = LogisticRegression()
        self.domain_classifier_ = clone(domain_classifier)
        y_domain = np.concatenate((len(features) * [0], len(features_target) * [1]))
        self.domain_classifier_.fit(
            np.concatenate((features, features_target)), y_domain
        )
        return

    def __call__(self, estimator, X, y):
        rng = check_random_state(self.random_state)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.33, random_state=rng
        )
        features_train = estimator.get_features(X_train)
        features_val = estimator.get_features(X_val)
        features_target = estimator.get_features(self.X_test)

        self._fit_adapt(features_train, features_target)
        N, N_target = len(features_train), len(features_target)
        predictions = self.domain_classifier_.predict_proba(features_val)
        weights = N / N_target * predictions[:, :1] / predictions[:, 1:]

        y_pred = estimator.predict_proba(X_val)
        error = self._loss_func(y_val, y_pred)
        assert weights.shape[0] == error.shape[0]

        weighted_error = weights * error
        cov = np.cov(np.concatenate((weighted_error, weights), axis=1), rowvar=False)[
            0
        ][1]
        var_w = np.var(weights, ddof=1)
        eta = -cov / var_w
        return self._sign * (np.mean(weighted_error) + eta * np.mean(weights) - eta)
