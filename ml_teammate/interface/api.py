
# ml_teammate/interface/api.py

from ml_teammate.learners.lightgbm_learner import get_lightgbm_learner
from ml_teammate.search.optuna_search import OptunaSearcher
from ml_teammate.automl.controller import AutoMLController
from ml_teammate.automl.callbacks import LoggerCallback

class MLTeammate:
    def __init__(self, task="classification", n_trials=5):
        self.task = task
        self.n_trials = n_trials

        # Prepare learner dictionary
        self.learners = {
            "lightgbm": get_lightgbm_learner
        }

        # Define config spaces per learner
        self.config_spaces = {
            "lightgbm": {
                "max_depth": (3, 8),
                "learning_rate": (0.01, 0.3),
                "n_estimators": (50, 300)
            }
        }

        # Create searcher
        self.searcher = OptunaSearcher(self.config_spaces)

        # Create logger callback
        self.logger_callback = LoggerCallback()

        # Create controller
        self.controller = AutoMLController(
            learners=self.learners,
            searcher=self.searcher,
            config_space=self.config_spaces,
            task=self.task,
            n_trials=self.n_trials,
            logger_callback=self.logger_callback
        )

    def fit(self, X, y):
        self.controller.fit(X, y)

    def predict(self, X):
        return self.controller.predict(X)
