
# ml_teammate/learners/lightgbm_learner.py

from lightgbm import LGBMClassifier

class LightGBMLearner:
    def __init__(self, config: dict):
        self.model = LGBMClassifier(**config)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

def get_lightgbm_learner(config: dict):
    return LightGBMLearner(config)

