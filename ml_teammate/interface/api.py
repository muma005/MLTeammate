
# ml_teammate/interface/api.py

from ml_teammate.learners.lightgbm_learner import get_lightgbm_learner
from ml_teammate.search.optuna_search import OptunaSearcher
from ml_teammate.automl.controller import AutoMLController
from ml_teammate.automl.callbacks import LoggerCallback
from ml_teammate.experiments.mlflow_helper import MLflowHelper

class MLTeammate:
    def __init__(self,
                 task: str = "classification",
                 n_trials: int = 5,
                 enable_mlflow: bool = False):
        self.task = task
        self.n_trials = n_trials

        # Learner registry
        self.learners = {
            "lightgbm": get_lightgbm_learner
        }

        # Config spaces per learner
        self.config_spaces = {
            "lightgbm": {
                "max_depth": {"type": "int", "bounds": [3, 8]},
                "learning_rate": {"type": "float", "bounds": [0.01, 0.3]},
                "n_estimators": {"type": "int", "bounds": [50, 300]},
            }
        }

        # Initialize searcher
        self.searcher = OptunaSearcher(self.config_spaces)

        # Initialize logger callbacks
        self.logger = LoggerCallback(use_mlflow=enable_mlflow)

        # Initialize MLflow if needed
        mlflow_helper = MLflowHelper() if enable_mlflow else None

        # Initialize controller
        self.controller = AutoMLController(
            learners=self.learners,
            searcher=self.searcher,
            config_space=self.config_spaces,
            task=self.task,
            n_trials=self.n_trials,
            callbacks=[self.logger],
            mlflow_helper=mlflow_helper
        )

    def fit(self, X, y):
        self.controller.fit(X, y)

    def predict(self, X):
        return self.controller.predict(X)
