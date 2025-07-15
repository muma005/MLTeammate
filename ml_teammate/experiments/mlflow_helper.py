import mlflow

class MLflowHelper:
    def __init__(self, experiment_name: str = "mlteammate_experiment"):
        mlflow.set_experiment(experiment_name)
        self.run = None

    def start_run(self, run_name: str = None):
        self.run = mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict):
        for k, v in params.items():
            mlflow.log_param(k, v)

    def log_metrics(self, metrics: dict, step: int = None):
        for k, v in metrics.items():
            mlflow.log_metric(k, v, step=step)

    def end_run(self):
        mlflow.end_run()
