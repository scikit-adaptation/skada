from abc import ABC, abstractmethod

import scipy
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression

from ..utils import Kernel


class BaseMetricDA(ABC):
    """
    Base class for DA metrics.
    """
    def __init__(self):
        pass
    
    @abstractmethod
    def fit(self, Xs, Xt, Ws=None, Wt=None):
        """Fit ..."""
    
    @abstractmethod
    def eval(self):
        """Eval ..."""
    
    @abstractmethod
    def gradient(self, variable):
        """Gradient ..."""
        raise NotImplementedError("Gradient is not implemented for this metric.")
        
    def update_weights(self, Ws, Wt):
        self.Ws_ = Ws
        self.Wt_ = Wt
        self.fit(self.Xs_, self.Xt_, Ws, Wt)
        
    def __call__(self, Xs, Xt, Ws=None, Wt=None):
        self.fit( Xs, Xt, Ws, Wt)
        return self.eval()
    
    
class MaximumMeanDiscrepancy(BaseMetricDA):
    """
    MMD ...
    """
    def __init__(self,
                 kernel="rbf",
                 gamma=None,
                 degree=3,
                 coef0=1,
                 biased=True):
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.biased = biased
        if self.kernel is None and not self.biased:
            raise ValueError("`biased` argument cannot be set to False "
                             "when `kernel` is `None`.")
        
    def fit(self, Xs, Xt, Ws=None, Wt=None):
        """
        Fit MMD ...
        """
        if self.kernel is None:
            self.Kss_ = Xs
            self.Kst_ = Xt
        else:
            self.kernel_ = Kernel(kernel=self.kernel,
                                  gamma=self.gamma,
                                  degree=self.degree,
                                  coef0=self.coef0)
            self.kernel_.fit(Xs)
            self.Kss_ = self.kernel_.predict(Xs)
            self.Kst_ = self.kernel_.predict(Xt).T
            if not self.biased:
                self.kernel_.fit(Xt)
                self.Ktt_ = self.kernel_.predict(Xt)
        self.Ws_ = Ws
        self.Wt_ = Wt
        return self
        
    def eval(self):
        if self.Ws_ is not None:
            Kss = self.Ws_ @ self.Kss_ @ self.Ws_
            Kss /= self.Kss_.shape[0]**2
            Kst = self.Ws_ @ self.Kst_
            Kst /= self.Kss_.shape[0]
        else:
            Kss = self.Kss_.mean()
            Kst = self.Kst_.mean(axis=0)
        
        if self.Wt_ is not None:
            Kst = Kst @ self.Wt_
            Kst /= self.Kst_.shape[1]
        else:
            Kst = Kst.mean()
            
        mmd = Kss - 2 * Kst
        
        if not self.biased:
            if self.Wt_ is not None:
                mmd += self.Wt_ @ self.Ktt_ @ self.Wt_
            else:
                mmd += np.mean(self.Ktt_) 
        return mmd
    
    def gradient(self):
        return self.Kss_.dot(self.Ws_.ravel()) - np.ravel(self.Kst_.sum(1))
    
    def update_weights(self, Ws, Wt):
        self.Ws_ = Ws
        self.Wt_ = Wt
    
    
def _covariances(X, W):
    """
    Compute covariance matrixes (for Discrepancy + CorrelationDifference)
    """
    if W is not None:
        C = X.T @ np.diag(W**2) @ X
    else:
        C = X.T @ X
    C /= X.shape[0]
    return C
    

class LinearDiscrepancy(BaseMetricDA):
    """
    LinearDiscrepancy ...
    """
    def __init__(self,
                 intercept=False):
        self.intercept = intercept
    
    def fit(self, Xs, Xt, Ws=None, Wt=None):
        """
        Fit Discrepancy ...
        """
        self.Xs_ = Xs
        self.Xt_ = Xt
        self.Ws_ = Ws
        self.Wt_ = Wt
        
        if self.intercept:
            Xs_ = np.concatenate((np.ones((Xs.shape[0], 1)), Xs), axis=1)
            Xt_ = np.concatenate((np.ones((Xt.shape[0], 1)), Xt), axis=1)
        else:
            Xs_ = Xs
            Xt_ = Xt
        
        self.Cs_ = _covariances(Xs_, Ws)
        self.Ct_ = _covariances(Xt_, Wt)
        
        eigen = scipy.linalg.eig(self.Cs_ - self.Ct_)
        self.max_eigval_ = np.abs(eigen[0]).max()
        self.sign_ = np.sign(np.real(eigen[0][np.abs(eigen[0]).argmax()]))
        self.max_eigvect_ = np.real(eigen[1][:, np.abs(eigen[0]).argmax()])
        return self
        
    def eval(self):
        return self.max_eigval_
  
    def gradient(self):
        if self.intercept:
            Xs_ = np.concatenate((np.ones((self.Xs_.shape[0], 1)), self.Xs_), axis=1)
        else:
            Xs_ = self.Xs_
        return 2 * self.Ws_ * self.sign_ * (Xs_ @ self.max_eigvect_)**2
    
    
class KernelDiscrepancy(LinearDiscrepancy):
    """
    KernelDiscrepancy ...
    """
    def __init__(self,
                 kernel="linear",
                 gamma=None,
                 degree=3,
                 coef0=1,
                 n_centers=None,
                 intercept=False):
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.n_centers = n_centers
        super().__init__(intercept=intercept)
        
    def fit(self, Xs, Xt, Ws=None, Wt=None):
        """
        Fit KernelDiscrepancy ...
        """
        if self.kernel is not None:
            self.kernel_ = Kernel(kernel=self.kernel,
                                  gamma=self.gamma,
                                  degree=self.degree,
                                  coef0=self.coef0,
                                  n_centers=self.n_centers)
            if scipy.sparse.issparse(Xs):
                X = scipy.sparse.vstack((Xs, Xt))
            else:
                X = np.concatenate((Xs, Xt))
            self.kernel_.fit(X)
            Xs = self.kernel_.predict(Xs)
            Xt = self.kernel_.predict(Xt)
        super().fit(Xs, Xt, Ws, Wt)
        return self
    
    def update_weights(self, Ws, Wt):
        self.Ws_ = Ws
        self.Wt_ = Wt
        super().fit(self.Xs_, self.Xt_, Ws, Wt)
        
        
class CorrelationDifference(BaseMetricDA):
    
    def __init__(self,
                 intercept=False):
        self.intercept = intercept
    
    def fit(self, Xs, Xt, Ws=None, Wt=None):
        """
        Fit CorrelationDifference ...
        """
        self.Xs_ = Xs
        self.Xt_ = Xt
        self.Ws_ = Ws
        self.Wt_ = Wt
        
        if self.intercept:
            Xs_ = np.concatenate((np.ones((Xs.shape[0], 1)), Xs), axis=1)
            Xt_ = np.concatenate((np.ones((Xt.shape[0], 1)), Xt), axis=1)
        else:
            Xs_ = Xs
            Xt_ = Xt
        
        self.Cs_ = _covariances(Xs_, Ws)
        self.Ct_ = _covariances(Xt_, Wt)
        return self
        
    def eval(self):
        return ((self.Cs_ - self.Ct_)**2).sum()
  
    def gradient(self):
        if self.intercept:
            Xs_ = np.concatenate((np.ones((self.Xs_.shape[0], 1)), self.Xs_), axis=1)
        else:
            Xs_ = self.Xs_
        D = (self.Cs_ - self.Ct_)
        R = np.stack([Xs_[[j], :].T @ Xs_[[j], :]
                      for j in range(Xs_.shape[0])], axis=-1)
        return 2 * self.Ws_ * (R * D[:, :, np.newaxis]).sum(axis=(0, 1))
    
    
class HDivergence(BaseMetricDA):
    """
    HDivergence
    """
    
    def __init__(self,
                 classifier=None,
                 n_splits=3,
    ):
        self.classifier = classifier
        self.n_splits = n_splits
        
    # TODO : balance split between src and tgt +
    # balance split according to Ws and Wt
    def fit(self, Xs, Xt, Ws=None, Wt=None, **fit_params):
        X = np.concatenate((Xs, Xt))
            
        y = np.zeros(X.shape[0])
        y[:Xs.shape[0]] = 1
        
        if Ws is not None:
            if Wt is not None:
                sample_weight = np.concatenate((Ws, Wt))
            else:
                sample_weight = np.concatenate((Ws, np.ones(Xt.shape[0])))
        else:
            if Wt is not None:
                sample_weight = np.concatenate((np.ones(Xs.shape[0]), Wt))
            else:
                sample_weight = None
                
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight
        
        if hasattr(self.classifier, "predict_proba"):
            method = "predict_proba"
        elif hasattr(self.classifier, "decision_function"):
            method = "decision_function"
        else:
            method = "predict"
        
        if self.classifier is None:
            classifier = LogisticRegression()
            method = "decision_function"
        else:
            classifier = self.classifier
        
        self.y_pred_ = cross_val_predict(
            classifier,
            X, y,
            cv=self.n_splits,
            method=method,
            fit_params=fit_params,
        )
        
        if method == "decision_function":
            self.y_pred_ = 1. / (1. + np.exp(-self.y_pred_))
        if method == "predict_proba":
            self.y_pred_ = self.y_pred_[:, 1]
        
        self.Xs_ = Xs
        self.Xt_ = Xt
        self.y_true_ = y
        self.sample_weight_ = sample_weight
        return self
    
    # TODO : Possibility to use other metrics than log loss
    def eval(self):
        return -log_loss(self.y_true_, self.y_pred_, labels=[0, 1],
                        sample_weight=self.sample_weight_)
    
    def gradient(self):
        return np.log(self.y_pred_[self.y_true_==1]+1e-15)
    
        
        
class RelativePearsonDivergence(BaseMetricDA):
    
    def __init__(self, alpha=0.1):
        self.alpha = alpha
    
    def fit(self, Xs, Xt, Ws=None, Wt=None):
        self.Xs_ = None
        self.Xt_ = None
        self.Ws_ = Ws
        self.Wt_ = Wt
    
    def eval(self):
        pe = -0.5 * self.alpha * (self.Ws_**2).mean()
        pe += -0.5 * (1-self.alpha) * (self.Wt_**2).mean()
        pe += self.Ws_.mean()
        pe -= 0.5
        return pe
    
    def gradient(self):
        grad = -self.alpha * self.Ws_
        grad -= (1-self.alpha) * self.Wt_
        grad += 1.
        return grad