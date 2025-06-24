
# ml_teammate/interface/api.py

from ml_teammate.automl.controller import AutoMLController
from ml_teammate.learners.lightgbm_learner import get_lightgbm_learner
from ml_teammate.search.optuna_search import OptunaSearcher

class MLTeammate:
    def __init__(self, task="classification", n_trials=10):
        # Learner registry
        self.learners = {"lightgbm": get_lightgbm_learner}

        # Per-learner config spaces
        config_spaces = {
            "lightgbm": {
                "max_depth": {"type": "int", "bounds": [2, 8]},
                "n_estimators": {"type": "int", "bounds": [10, 100]},
                "learning_rate": {"type": "float", "bounds": [0.01, 0.3]},
            }
        }

        # Searcher and controller
        self.searcher = OptunaSearcher(config_spaces)
        self.controller = AutoMLController(
            learners=self.learners,
            searcher=self.searcher,
            task=task,
            n_trials=n_trials
        )

    def fit(self, X, y):
        self.controller.fit(X, y)

    def predict(self, X):
        return self.controller.predict(X)
