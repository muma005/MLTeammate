import uuid
from ml_teammate.utils.metrics import evaluate

class AutoMLController:
    def __init__(
        self,
        learners: dict,
        searcher,
        config_space: dict,
        task: str = "classification",
        n_trials: int = 10,
        callbacks: list = None,
        mlflow_helper = None
    ):
        self.learners = learners
        self.searcher = searcher
        self.config_space = config_space
        self.task = task
        self.n_trials = n_trials
        self.callbacks = callbacks or []
        self.mlflow = mlflow_helper
        self.best_score = None
        self.best_model = None

    def fit(self, X, y):
        if self.mlflow:
            self.mlflow.start_run()

        for i in range(self.n_trials):
            trial_id = str(uuid.uuid4())
            learner_name = next(iter(self.learners))  # single learner for now

            config = self.searcher.suggest(trial_id, learner_name)
            model = self.learners[learner_name](config)
            model.fit(X, y)

            preds = model.predict(X)
            score = evaluate(y, preds, task=self.task)
            print(f"Trial {i+1}/{self.n_trials} — {learner_name} score={score:.4f}")

            self.searcher.report(trial_id, score)

            is_best = self.best_score is None or score < self.best_score
            if is_best:
                self.best_score = score
                self.best_model = model

            for cb in self.callbacks:
                cb.on_trial_end(trial_id, config, score, is_best)

            if self.mlflow:
                self.mlflow.log_params(config)
                self.mlflow.log_metrics({"score": score}, step=i+1)

        if self.mlflow:
            self.mlflow.end_run()

    def predict(self, X):
        if self.best_model is None:
            raise RuntimeError("Call fit() before predict()")
        return self.best_model.predict(X)
