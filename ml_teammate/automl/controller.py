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
        # Prepare experiment configuration
        experiment_config = {
            "task": self.task,
            "n_trials": self.n_trials,
            "cv": self.cv,
            "learners": list(self.learners.keys()),
            "data_shape": X.shape
        }
        
        # Notify callbacks of experiment start
        for cb in self.callbacks:
            cb.on_experiment_start(experiment_config)
        
        if self.mlflow:
            self.mlflow.start_run()

        for i in range(self.n_trials):
            trial_id = str(uuid.uuid4())
            start_time = time.time()
            learner_name = "xgboost"  # force using xgboost for a quick test

            try:
                config = self.searcher.suggest(trial_id, learner_name)
                
                # Notify callbacks of trial start
                for cb in self.callbacks:
                    cb.on_trial_start(trial_id, config)
                
                model = self.learners[learner_name](config)

                if self.cv:
                    # Cross-validation path
                    preds = cross_val_predict(model, X, y, cv=self.cv)
                    model.fit(X, y)  # still fit full model for .predict()
                else:
                    model.fit(X, y)  # simple fallback without pruning
                    preds = model.predict(X)
                
                duration = time.time() - start_time
                score = evaluate(y, preds, task=self.task)

                self.searcher.report(trial_id, score)

                is_best = self.best_score is None or score > self.best_score
                if is_best:
                    self.best_score = score
                    self.best_model = model

                # Notify callbacks of trial end
                for cb in self.callbacks:
                    cb.on_trial_end(trial_id, config, score, is_best)

                if self.mlflow:
                    self.mlflow.log_params(config)
                    self.mlflow.log_metrics({"score": score}, step=i+1)

            except Exception as e:
                print(f"[Warning] Trial {i+1} failed: {e}")
                continue

        # Notify callbacks of experiment end
        for cb in self.callbacks:
            cb.on_experiment_end(self.best_score, self.searcher.get_best())

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
    
