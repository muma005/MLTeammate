
# ml_teammate/search/optuna_search.py

import optuna

class OptunaSearcher:
    def __init__(self, config_spaces, direction="minimize"):
        self.config_spaces = config_spaces  # Dict[learner_name] -> search space
        self.study = optuna.create_study(direction=direction)
        self.trials = {}  # trial_id -> optuna trial

    def suggest(self, trial_id, learner_name):
        def objective(trial):
            self.trials[trial_id] = trial
            config = {}
            for param, space in self.config_spaces[learner_name].items():
                if space["type"] == "int":
                    config[param] = trial.suggest_int(param, *space["bounds"])
                elif space["type"] == "float":
                    config[param] = trial.suggest_float(param, *space["bounds"])
                elif space["type"] == "categorical":
                    config[param] = trial.suggest_categorical(param, space["choices"])
            return 0.0  # dummy score — real eval is external
        # Register dummy trial for now (score will come later)
        self.study.optimize(objective, n_trials=1, catch=(Exception,))
        return self.study.trials[-1].params

    def report(self, trial_id, score):
        trial = self.trials.get(trial_id)
        if trial is not None:
            trial._set_user_attr("score", score)
            trial.value = score  # manually set trial result

    def get_best_config(self):
        return self.study.best_trial.params
