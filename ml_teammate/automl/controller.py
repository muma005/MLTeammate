from ml_teammate.utils.metrics import evaluate
import uuid

class AutoMLController:
    def __init__(self, learners, search, config_space, task="classification", n_trials=10, callbacks=None, mlflow_helper=None):
        self.learners = learners
        self.search = search
        self.config_space = config_space
        self.task = task
        self.n_trials = n_trials
        self.callbacks = callbacks if callbacks is not None else []
        self.mlflow_helper = mlflow_helper
        self.best_score = None
        self.best_model = None

    def fit(self, X, y):
        if self.mlflow_helper:
            self.mlflow_helper.start_run()

        for trial_num in range(self.n_trials):
            trial_id = str(uuid.uuid4())
            learner_name = "lightgbm"  # You can later generalize this

            config = self.search.suggest(trial_id, learner_name)
            model = self.learners[learner_name](config)
            model.fit(X, y)

            y_pred = model.predict(X)
            score = evaluate(y, y_pred, task=self.task)

            print(f"Trial {trial_num + 1}/{self.n_trials} — {learner_name} score={score:.4f}")

            self.search.report(trial_id, score)

            # Callbacks after each trial
            for callback in self.callbacks:
                callback.on_trial_end(trial_num + 1, score, config)

            # MLflow log
            if self.mlflow_helper:
                self.mlflow_helper.log_params(config)
                self.mlflow_helper.log_metrics({"score": score}, step=trial_num + 1)

            # Track best
            if (self.best_score is None) or (score < self.best_score):
                self.best_score = score
                self.best_model = model

        if self.mlflow_helper:
            self.mlflow_helper.end_run()

    def predict(self, X):
        if self.best_model is None:
            raise ValueError("You must call .fit() before .predict()")
        return self.best_model.predict(X)
