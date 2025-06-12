
from ml_teammate.automl.automl import AutoML

class MLTeammate:
    def __init__(self):
        self.automl = AutoML()

    def fit(self, X, y):
        self.automl.fit(X, y)

    def predict(self, X):
        return self.automl.predict(X)
