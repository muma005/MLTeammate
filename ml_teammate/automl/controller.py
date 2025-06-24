# ml_teammate/automl/controller.py

import uuid
from ml_teammate.utils.metrics import evaluate

class AutoMLController:
    def __init__(self, learners, searcher, task="classification", n_trials=10):
        self.learners = learners            # e.g. {"lightgbm": get_lightgbm_learner}
        self.searcher = searcher            # instance of OptunaSearcher
        self.task = task
        self.n_trials = n_trials
        self.best_score = None
        self.best_model = None

    def fit(self, X, y):
        for i in range(self.n_trials):
            tid = str(uuid.uuid4())
            learner_name, learner_fn = next(iter(self.learners.items()))
            config = self.searcher.suggest(tid, learner_name)

            model = learner_fn(**config)
            model.fit(X, y)
            preds = model.predict(X)
            score = evaluate(y, preds, task=self.task)

            print(f"Trial {i+1}/{self.n_trials} — {learner_name} score={score:.4f}")

            self.searcher.report(tid, score)

            if self.best_score is None or score < self.best_score:
                self.best_score = score
                self.best_model = model

        return self.best_model

    def predict(self, X):
        if self.best_model is None:
            raise RuntimeError("Must call fit() before predict()")
        return self.best_model.predict(X)
