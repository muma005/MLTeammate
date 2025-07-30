# ml_teammate/experiments/mlflow_helper.py
# ml_teammate/experiments/mlflow_helper.py
import mlflow
from contextlib import contextmanager

class MLflowHelper:
    def __init__(self, experiment_name: str = "mlteammate_experiment", tracking_uri: str = None):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._run = None

    @contextmanager
    def start_run(self, run_name: str = None):
        """Proper context manager implementation"""
        self._run = mlflow.start_run(run_name=run_name)
        try:
            yield self
        finally:
            self.end_run()

    def log_params(self, params: dict):
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict, step: int = None):
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str):
        mlflow.log_artifact(local_path)

    def end_run(self):
        if self._run:
            mlflow.end_run()
            self._run = None