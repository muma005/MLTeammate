# ml_teammate/automl/controller.py

import uuid
from ml_teammate.utils.metrics import evaluate


class AutoMLController:
    """
    Orchestrates AutoML trials:
      - asks searcher for configs
      - trains learners
      - evaluates and reports back
      - tracks the best model
    """

    def __init__(self, learners: dict, searcher, task: str = "classification", n_trials: int = 10):
        """
        learners: dict mapping learner_name to a callable (model constructor)
        searcher: an instance of OptunaSearch or any Searcher interface
        """
        self.learners = learners
        self.searcher = searcher
        self.task = task
        self.n_trials = n_trials
        self.best_score = None
        self.best_model = None

    def fit(self, X, y):
        for trial_num in range(self.n_trials):
            trial_id = str(uuid.uuid4())
            # Here, we're using a single learner; later you can loop learners
            learner_name = "lightgbm"

            # 1) Get hyperparameters
            config = self.searcher.suggest(trial_id, learner_name)

            # 2) Instantiate and train model
            model = self.learners[learner_name](**config)
            model.fit(X, y)

            # 3) Evaluate
            y_pred = model.predict(X)
            score = evaluate(y, y_pred, task=self.task)
            print(f"[Trial {trial_num + 1}/{self.n_trials}] Score={score:.4f} Config={config}")

            # 4) Report back
            self.searcher.report(trial_id, score)

            # 5) Track best
            if (self.best_score is None) or (score < self.best_score):
                self.best_score = score
                self.best_model = model

        # After all trials, best_model is ready
        return self.best_model

    def predict(self, X):
        if self.best_model is None:
            raise RuntimeError("Must call fit() before predict()")
        return self.best_model.predict(X)


