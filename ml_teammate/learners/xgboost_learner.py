from xgboost import XGBClassifier

class XGBoostLearner:
    def __init__(self, config):
        self.model = XGBClassifier(**config)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

def get_xgboost_learner(config):
    return XGBoostLearner(config)
