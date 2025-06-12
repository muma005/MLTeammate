from ml_teammate.search.optuna_search import suggest
from ml_teammate.learners.lightgbm_learner import LightGBMLearner

class Controller:
    def select_next_config(self, X, y):
        return suggest(X, y)  # Dummy hyperparameters

    def train(self, config, X, y):
        learner = LightGBMLearner(config)
        learner.fit(X, y)
        return learner
