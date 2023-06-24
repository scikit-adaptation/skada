import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels, KERNEL_PARAMS


class Kernel:
    """
    Class for computing kernel distances
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
        
        if not kernel is None and not kernel in KERNEL_PARAMS:
            list_kernels = [k for k in KERNEL_PARAMS] + [None]
            raise ValueError("`kernel` argument should be included in %s,"
                             " got '%s'"%(str(list_kernels), str(kernel)))
        
    
    def fit(self, X):
        """
        Centers selection
        """
        if self.n_centers is None or self.n_centers >= X.shape[0]:
            self.centers_ = X
            self.centers_index_ = np.arange(X.shape[0])
            
        else:
            self.centers_index_ = np.random.choice(X.shape[0],
                                                   size=self.n_centers,
                                                   replace=False)
            self.centers_ = X[self.centers_index_]
            
    def predict(self, X):
        """
        Compute Kernel pairwise distances
        """
        Xt = pairwise_kernels(X, self.centers_,
                              metric=self.kernel,
                              filter_params=True,
                              **self.__dict__)
        return Xt