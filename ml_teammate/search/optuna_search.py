


import optuna

class OptunaSearch:
    def __init__(self, config_space):
        self.config_space = config_space
        self.study = optuna.create_study(direction="minimize")
        self.trials = {}

    def suggest(self, trial_id, learner_name):
        trial = self.study.ask()
        self.trials[trial_id] = trial
        config = self.config_space.sample(trial, learner_name)
        return config

    def report(self, trial_id, score):
        trial = self.trials[trial_id]
        self.study.tell(trial, score)

    def get_best_config(self):
        return self.study.best_trial.params
