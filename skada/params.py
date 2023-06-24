import numpy as np
from .utils import Kernel


class KernelWeighting:
    """
    Kernel parameterization of importance weights
    """
    
    def __init__(self,
                 kernel="rbf",
                 gamma=None,
                 degree=3,
                 coef0=1,
                 n_centers=None
    ):
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.n_centers = n_centers
        
    
    def fit(self, X, Xs=None, Xt=None):
        """
        Fit kernel weighting
        """
        self.kernel_ = Kernel(kernel=self.kernel,
                              gamma=self.gamma,
                              degree=self.degree,
                              coef0=self.coef0,
                              n_centers=self.n_centers)
        self.kernel_.fit(X)
        if self.n_centers is None:
            self.theta_ = np.ones(X.shape[0])
        else:
            self.theta_ = np.ones(min(X.shape[0], self.n_centers))
        if Xs is not None:
            self.Ks_ = self.kernel_.predict(Xs)
        if Xt is not None:
            self.Kt_ = self.kernel_.predict(Xt)
        
    def predict(self, X="src"):
        """
        Predict weight
        """
        if isinstance(X, str):
            if X == "src":
                return self.Ks_ @ self.theta_
            elif X == "tgt":
                return self.Kt_ @ self.theta_
        else:
            return self.kernel_.predict(X) @ self.theta_
            
    def update(self, theta):
        self.theta_ = theta
        
    def gradient(self, X="src"):
        if isinstance(X, str):
            if X == "src":
                return self.Ks_.T
            elif X == "tgt":
                return self.Kt_.T
        else:
            return self.kernel_.predict(X).T
        
        
        
class LinearWeighting:
    """
    Linear parameterization of importance weights
    """
    
    def __init__(self):
        pass
        
    
    def fit(self, X, Xs=None, Xt=None):
        """
        Fit linear weighting
        """
        self.theta_ = np.ones(X.shape[1])
        self.Xs_ = Xs
        self.Xt_ = Xt
        
    def predict(self, X="src"):
        """
        Predict weight
        """
        if isinstance(X, str):
            if X == "src":
                return self.Xs_ @ self.theta_
            elif X == "tgt":
                return self.Xt_ @ self.theta_
        else:
            return X @ self.theta_
            
    def update(self, theta):
        self.theta_ = theta
        
    def gradient(self, X="src"):
        if isinstance(X, str):
            if X == "src":
                return self.Xs_.T
            elif X == "tgt":
                return self.Xt_.T
        else:
            return X.T
    
    
class DirectWeighting:
    """
    Direct parameterization of importance weights
    """
    
    def __init__(self):
        pass
        
    
    def fit(self, X, Xs=None, Xt=None):
        """
        Fit kernel weighting
        """
        self.theta_ = np.ones(X.shape[0])
        self.Xs_ = Xs
        self.Xt_ = Xt
        
    def predict(self, X="src"):
        """
        Predict weight
        """
        if isinstance(X, str):
            if X == "src":
                return self.theta_
            elif X == "tgt":
                return None
        else:
            return None
            
    def update(self, theta):
        self.theta_ = theta
        
    def gradient(self, X="src"):
        if isinstance(X, str):
            if X == "src":
                return np.eye(self.theta_.shape[0])
            elif X == "tgt":
                return None
        else:
            return None
    
    
