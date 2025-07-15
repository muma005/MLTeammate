
# ml_teammate/interface/api.py

from ml_teammate.automl.controller import AutoMLController
from ml_teammate.learners.lightgbm_learner import get_lightgbm_learner
from ml_teammate.search.optuna_search import OptunaSearcher
from ml_teammate.automl.callbacks import LoggerCallback
from ml_teammate.experiments.mlflow_helper import MLflowHelper

class MLTeammate:
    def __init__(self, task="classification", n_trials=10, enable_mlflow=False):
        self.task = task
        self.n_trials = n_trials
        self.enable_mlflow = enable_mlflow

        # Setup learners
        self.learners = {
            "lightgbm": get_lightgbm_learner
        }

        # Setup config space
        self.config_spaces = {
            "lightgbm": {
                "max_depth": (3, 8),
                "learning_rate": (0.01, 0.2),
                "n_estimators": (50, 200)
            }
        }

        # Setup searcher
        self.searcher = OptunaSearcher(self.config_spaces)

        # Setup logger callback
        self.logger_callback = LoggerCallback()

        # Setup mlflow if enabled
        self.mlflow_helper = MLflowHelper(task_name=task) if enable_mlflow else None

        # Setup controller
        self.controller = AutoMLController(
            learners=self.learners,
            searcher=self.searcher,
            config_space=self.config_spaces,
            task=self.task,
            n_trials=self.n_trials,
            callbacks=[self.logger_callback],
            mlflow_helper=self.mlflow_helper
        )

    def fit(self, X, y):
        self.controller.fit(X, y)

    def predict(self, X):
        return self.controller.predict(X)
