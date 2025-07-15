class LoggerCallback:
    def __init__(self, use_mlflow: bool = False):
        self.use_mlflow = use_mlflow
        if use_mlflow:
            from ml_teammate.experiments.mlflow_helper import MLflowHelper
            self.mlflow = MLflowHelper()
        else:
            self.mlflow = None

    def on_trial_end(self, trial_id: str, config: dict, score: float, is_best: bool):
        print(f"[Logger] Trial {trial_id} ended — Score: {score:.4f}")
        print(f"         Config: {config}")
        if is_best:
            print("         ✅ New best model!")
        if self.use_mlflow:
            self.mlflow.log_params(config)
            self.mlflow.log_metrics({"score": score})
