
from ml_teammate.automl.controller import AutoMLController
from ml_teammate.learners.lightgbm_learner import get_lightgbm_learner
from ml_teammate.search.optuna_search import OptunaSearcher


class MLTeammate:
    def __init__(self, task="classification", n_trials=10):
        self.task = task
        self.n_trials = n_trials

        # Define the learner registry
        self.learners = {
            "lightgbm": get_lightgbm_learner,
        }

        # Define the search space per learner
        config_spaces = {
            "lightgbm": {
                "max_depth": {"type": "int", "bounds": [2, 8]},
                "n_estimators": {"type": "int", "bounds": [10, 100]},
                "learning_rate": {"type": "float", "bounds": [0.01, 0.2]},
            }
        }

        # Initialize search algorithm
        self.search = OptunaSearcher(config_spaces)

        # Controller ties everything together
        self.controller = AutoMLController(
            learners=self.learners,
            search=self.search,
            config_space=config_spaces,
            task=self.task,
            n_trials=self.n_trials
        )

    def fit(self, X, y):
        self.controller.fit(X, y)

    def predict(self, X):
        return self.controller.predict(X)
