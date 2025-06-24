
# ml_teammate/learners/lightgbm_learner.py

from lightgbm import LGBMClassifier

class LightGBMLearner:
    def __init__(self, config):
        self.model = LGBMClassifier(**config)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

# ✅ This is the expected function interface for the controller/api
def get_lightgbm_learner(**config):
    return LightGBMLearner(config)
