
# ml_teammate/search/optuna_search.py

import optuna
from typing import Dict


class OptunaSearch:
    """
    A Searcher that uses Optuna to suggest and record hyperparameter trials.
    """

    def __init__(self, learner_name: str, direction: str = "minimize", seed: int = 42):
        self.learner_name = learner_name
        self.study = optuna.create_study(
            direction=direction,
            study_name=f"{learner_name}_search",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        self.trial_map = {}  # Map from trial_id to Optuna Trial

    def suggest(self, trial_id: str, learner_name: str) -> Dict:
        """
        Ask Optuna for a new trial and return a hyperparameter dict
        specific to the given learner_name.
        """
        trial = self.study.ask()
        self.trial_map[trial_id] = trial

        # Define per-learner search spaces here:
        if learner_name == "lightgbm":
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 256),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            }
        else:
            raise NotImplementedError(f"No config space defined for learner: {learner_name}")

        return params

    def report(self, trial_id: str, score: float):
        """
        Tell Optuna the result (score) of the trial identified by trial_id.
        """
        trial = self.trial_map.get(trial_id)
        if trial is None:
            raise ValueError(f"Unknown trial_id: {trial_id}")
        self.study.tell(trial, score)

    def get_best(self) -> Dict:
        """
        Return the best hyperparameter set found so far.
        """
        return self.study.best_trial.params

    def run_search(self, fit_fn, learner_name: str, n_trials: int = 10):
        """
        Optional helper: loops internally over n_trials, calling fit_fn(params) each time.
        fit_fn should accept a config dict and return a scalar score.
        """
        for i in range(n_trials):
            tid = f"trial_{i}"
            params = self.suggest(tid, learner_name)
            score = fit_fn(params)
            self.report(tid, score)
        return self.get_best()
