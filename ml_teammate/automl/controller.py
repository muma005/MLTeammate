#controller.py 
import uuid
import time 
from sklearn.model_selection import cross_val_predict  # ✅ CV support
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
        mlflow_helper = None,
        cv: int = None  # ✅ added
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
        self.cv = cv  # ✅ store cross-validation folds

    def fit(self, X, y):
        if self.mlflow:
            self.mlflow.start_run()

        for i in range(self.n_trials):
            trial_id = str(uuid.uuid4())
            start_time = time.time()  # ✅ Start timer
            learner_name = "xgboost"  # force using xgboost for a quick test

            try:
                config = self.searcher.suggest(trial_id, learner_name)
                model = self.learners[learner_name](config)

                if self.cv:
                    # ✅ Cross-validation path
                    preds = cross_val_predict(model, X, y, cv=self.cv)
                    model.fit(X, y)  # still fit full model for .predict()
                else:
                    # 🔧 Pruning logic temporarily disabled
                    # if hasattr(self.searcher, "get_pruning_callback"):
                    #     try:
                    #         pruning_cb = self.searcher.get_pruning_callback(trial_id, model, X, y)
                    #         model.fit(X, y, callbacks=[pruning_cb])
                    #     except Exception:
                    #         model.fit(X, y)
                    # else:
                    #     model.fit(X, y)
                    model.fit(X, y)  # simple fallback without pruning

                    preds = model.predict(X)
                duration = time.time() - start_time  # ✅ End timer
                score = evaluate(y, preds, task=self.task)
                print(f"Trial {i+1}/{self.n_trials} | Learner: {learner_name} | Score: {score:.4f} | Duration: {duration:.2f}s")
                print(f"Config: {config}")

                print(f"Trial {i+1}/{self.n_trials} — {learner_name} score={score:.4f}")

                self.searcher.report(trial_id, score)

                is_best = self.best_score is None or score > self.best_score
                if is_best:
                    self.best_score = score
                    self.best_model = model

                for cb in self.callbacks:
                    cb.on_trial_end(trial_id, config, score, is_best)

                if self.mlflow:
                    self.mlflow.log_params(config)
                    self.mlflow.log_metrics({"score": score}, step=i+1)

            except Exception as e:
                print(f"[Warning] Trial {i+1} failed: {e}")
                continue

        if self.mlflow:
            self.mlflow.end_run()

    def predict(self, X):
        if self.best_model is None:
            raise ValueError("No trained model found. Please call .fit() first and ensure at least one successful trial.")
        return self.best_model.predict(X) 

    def score(self, X, y):
        """
        Returns a task-appropriate score for the best model on given data.
        Equivalent to sklearn's .score() method.
        """
        preds = self.predict(X)
        return evaluate(y, preds, task=self.task)
    
