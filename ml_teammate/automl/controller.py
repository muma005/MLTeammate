from utils.metrics import evaluate
import uuid

class AutoMLController:
    def __init__(self, learners, searcher, config_space, task="classification", n_trials=10, logger_callback=None):
        self.learners = learners
        self.searcher = searcher
        self.config_space = config_space
        self.task = task
        self.n_trials = n_trials
        self.logger_callback = logger_callback
        self.best_score = None
        self.best_model = None
        self.best_config = None

    def fit(self, X, y):
        for trial_num in range(self.n_trials):
            trial_id = str(uuid.uuid4())
            learner_name = "lightgbm"  # ← you can change dynamically if supporting multiple learners

            # Suggest a new configuration
            config = self.searcher.suggest(trial_id, learner_name)

            # Initialize learner
            learner_class = self.learners[learner_name]
            learner = learner_class(config)
            learner.fit(X, y)

            # Evaluate
            y_pred = learner.predict(X)
            score = evaluate(y, y_pred, task=self.task)

            # Report score to searcher
            self.searcher.report(trial_id, score)

            # Check if best
            is_best = False
            if (self.best_score is None) or (score < self.best_score):
                self.best_score = score
                self.best_model = learner
                self.best_config = config
                is_best = True

            # Call logger callback
            if self.logger_callback:
                self.logger_callback.on_trial_end(trial_id, config, score, is_best)

            print(f"Trial {trial_num + 1}/{self.n_trials} — {learner_name} score={score:.4f}")

    def predict(self, X):
        if self.best_model is None:
            raise ValueError("You must call .fit() before .predict()")
        return self.best_model.predict(X)
