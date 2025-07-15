class LoggerCallback:
    def __init__(self, use_mlflow=False):
        self.use_mlflow = use_mlflow
        if use_mlflow:
            from ml_teammate.experiments.mlflow_helper import log_params, log_metrics
            self.log_params = log_params
            self.log_metrics = log_metrics
        else:
            self.log_params = None
            self.log_metrics = None

    def on_trial_end(self, trial_id, config, score, is_best=False):
        print(f"[Logger] Trial {trial_id} ended.")
        print(f"  Config: {config}")
        print(f"  Score: {score:.4f}")
        if is_best:
            print("  ✅ This is the new best model!")

        if self.use_mlflow and self.log_params and self.log_metrics:
            self.log_params(config)
            self.log_metrics({"score": score})
