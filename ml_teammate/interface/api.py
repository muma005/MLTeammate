
# ml_teammate/interface/api.py

from ml_teammate.automl.controller import AutoMLController
from ml_teammate.learners.lightgbm_learner import get_lightgbm_learner
from ml_teammate.search.optuna_search import OptunaSearcher
from ml_teammate.utils.metrics import evaluate

class MLTeammate:
    def __init__(self, task="classification", n_trials=10):
        self.task = task
        self.n_trials = n_trials

        # Define learners
        self.learners = {
            "lightgbm": get_lightgbm_learner,
        }

        # Search and config space setup
        self.search = OptunaSearcher()
        self.config_space = {
            "lightgbm": {
                "n_estimators": (10, 100),
                "max_depth": (3, 10),
                "learning_rate": (0.01, 0.3)
            }
        }

        # Setup controller
        self.controller = AutoMLController(
            learners=self.learners,
            search=self.search,
            config_space=self.config_space,
            task=self.task,
            n_trials=self.n_trials
        )

    def fit(self, X, y):
        self.controller.fit(X, y)

    def predict(self, X):
        return self.controller.predict(X)

    def log(self):
        print("Logging not yet implemented.")

