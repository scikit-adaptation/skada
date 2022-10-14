from sklearn.base import clone


class BaseDAEstimator():

    def __init__(
        self,
        base_estimator,
    ):
        self.base_estimator = base_estimator

    def fit(self, X, y, X_target, y_target=None):
        base_estimator = clone(self.base_estimator)
        self.fit_adapt(X, y, X_target)  # move X to target space
        Xt = self.transform_adapt(X)
        base_estimator.fit(Xt, y)
        self.base_estimator_ = base_estimator

    def predict(self, X):
        base_estimator = self.base_estimator_
        return base_estimator.predict(X)
