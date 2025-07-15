import mlflow

class MLflowHelper:
    def __init__(self, experiment_name="default_experiment"):
        mlflow.set_experiment(experiment_name)
        self.run = None

    def start_run(self, run_name=None):
        self.run = mlflow.start_run(run_name=run_name)

    def log_params(self, params):
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_metrics(self, metrics, step=None):
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    def end_run(self):
        mlflow.end_run()
