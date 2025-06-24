
# ml_teammate/search/optuna_search.py

import optuna

class OptunaSearcher:
    def __init__(self, config_spaces, direction="minimize", seed=42):
        self.config_spaces = config_spaces
        self.study = optuna.create_study(direction=direction,
                                         sampler=optuna.samplers.TPESampler(seed=seed))
        self.trials = {}

    def suggest(self, trial_id, learner_name):
        trial = self.study.ask()
        self.trials[trial_id] = trial
        space = self.config_spaces[learner_name]
        config = {}
        for name, spec in space.items():
            if spec["type"] == "int":
                config[name] = trial.suggest_int(name, *spec["bounds"])
            elif spec["type"] == "float":
                config[name] = trial.suggest_float(name, *spec["bounds"])
            elif spec["type"] == "categorical":
                config[name] = trial.suggest_categorical(name, spec["choices"])
        return config

    def report(self, trial_id, score):
        trial = self.trials.get(trial_id)
        if not trial:
            raise KeyError(f"Trial {trial_id} not registered")
        self.study.tell(trial, score)

    def get_best_config(self):
        return self.study.best_trial.params
