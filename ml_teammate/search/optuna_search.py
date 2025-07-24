
# ml_teammate/search/optuna_search.py

import optuna
from typing import Dict 
from optuna.trial import TrialState
#from optuna.integration import OptunaPruningCallback




class OptunaSearcher:
    def __init__(self, config_spaces: Dict[str, Dict]):
        """
        config_spaces: 
          {"learner_name": {"param": {"type": "int"/"float"/"categorical", "bounds"/"choices": [...]}, ...}}
        """
        self.config_spaces = config_spaces
        self.study = optuna.create_study(direction="minimize",
                                         sampler=optuna.samplers.TPESampler())
        self.trials = {}

    def suggest(self, trial_id: str, learner_name: str) -> Dict:
        trial = self.study.ask()
        self.trials[trial_id] = trial

        space = self.config_spaces[learner_name]
        config = {}
        for name, spec in space.items():
            t = spec["type"]
            if t == "int":
                low, high = spec["bounds"]
                config[name] = trial.suggest_int(name, low, high)
            elif t == "float":
                low, high = spec["bounds"]
                config[name] = trial.suggest_float(name, low, high)
            elif t == "categorical":
                config[name] = trial.suggest_categorical(name, spec["choices"])
        return config

    def report(self, trial_id: str, score: float):
        trial = self.trials.get(trial_id)
        if not trial:
            raise KeyError(f"Unknown trial_id: {trial_id}")
        self.study.tell(trial, score)
 # temporaliry disabled pruning callback
 #   def get_pruning_callback(self, trial_id: str, estimator, X, y):
 #       trial = self.trials.get(trial_id)
 #       if not trial:
 #           raise KeyError(f"Unknown trial_id: {trial_id}")

 #       return OptunaPruningCallback(trial, "accuracy")
    

    def get_best(self) -> Dict:
        return self.study.best_trial.params
