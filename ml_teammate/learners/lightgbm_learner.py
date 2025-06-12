
from lightgbm import LGBMClassifier

class LightGBMLearner:
    def __init__(self, config):
        self.model = LGBMClassifier(**config)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
