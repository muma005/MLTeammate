# ml_teammate/experiments/mlflow_helper.py
"""
Enhanced MLflow helper with nested runs support for proper experiment tracking.
Provides professional-grade experiment management with trial-level granularity.
"""

import mlflow
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Any, Optional

class MLflowHelper:
    """
    Enhanced MLflow helper for MLTeammate experiments.
    
    Features:
    - Nested runs for trial-level tracking
    - Proper experiment hierarchy
    - Rich metadata logging
    - Professional experiment structure
    """
    
    def __init__(self, experiment_name: str = "mlteammate_experiment", tracking_uri: str = None):
        """
        Initialize the MLflow helper.
        
        Args:
            experiment_name: Name for the MLflow experiment
            tracking_uri: Optional tracking URI (file, http, etc.)
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        self.parent_run = None
        self.current_trial_run = None
        self.experiment_start_time = None
    
    def start_experiment(self, run_name: str = None, experiment_config: Dict[str, Any] = None):
        """
        Start the main experiment run.
        
        Args:
            run_name: Optional name for the experiment run
            experiment_config: Configuration to log at experiment level
        """
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"experiment_{timestamp}"
        
        self.parent_run = mlflow.start_run(run_name=run_name)
        self.experiment_start_time = datetime.now()
        
        # Log experiment-level configuration
        if experiment_config:
            mlflow.log_params(experiment_config)
        
        # Log experiment metadata
        mlflow.set_tags({
            "experiment_type": "automl",
            "framework": "mlteammate",
            "start_time": self.experiment_start_time.isoformat()
        })
        
        return self
    
    def start_trial(self, trial_id: str, trial_number: int, trial_config: Dict[str, Any] = None):
        """
        Start a nested run for a specific trial.
        
        Args:
            trial_id: Unique identifier for the trial
            trial_number: Sequential trial number
            trial_config: Trial-specific configuration to log
        """
        if self.parent_run is None:
            raise ValueError("Must start experiment before starting trials")
        
        run_name = f"trial_{trial_number}_{trial_id[:8]}"
        self.current_trial_run = mlflow.start_run(run_name=run_name, nested=True)
        
        # Log trial metadata
        mlflow.set_tags({
            "trial_id": trial_id,
            "trial_number": trial_number,
            "parent_run_id": self.parent_run.info.run_id
        })
        
        # Log trial configuration if provided
        if trial_config:
            mlflow.log_params(trial_config)
        
        return self
    
    def log_trial_params(self, params: Dict[str, Any]):
        """
        Log parameters to the current trial run.
        
        Args:
            params: Dictionary of parameters to log
        """
        if self.current_trial_run is None:
            raise ValueError("No active trial run. Call start_trial() first.")
        mlflow.log_params(params)
    
    def log_trial_metrics(self, metrics: Dict[str, Any], step: int = None):
        """
        Log metrics to the current trial run.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number for the metrics
        """
        if self.current_trial_run is None:
            raise ValueError("No active trial run. Call start_trial() first.")
        mlflow.log_metrics(metrics, step=step)
    
    def log_trial_artifact(self, local_path: str, artifact_path: str = None):
        """
        Log an artifact to the current trial run.
        
        Args:
            local_path: Path to the local file to log
            artifact_path: Optional path within the run's artifact directory
        """
        if self.current_trial_run is None:
            raise ValueError("No active trial run. Call start_trial() first.")
        mlflow.log_artifact(local_path, artifact_path)
    
    def end_trial(self):
        """End the current trial run."""
        if self.current_trial_run:
            mlflow.end_run()
            self.current_trial_run = None
    
    def log_experiment_summary(self, best_score: float, best_config: Dict[str, Any], 
                             total_trials: int, experiment_duration: float = None):
        """
        Log final results to the parent run.
        
        Args:
            best_score: Best score achieved during the experiment
            best_config: Best configuration found
            total_trials: Total number of trials completed
            experiment_duration: Duration of the experiment in seconds
        """
        if self.parent_run is None:
            raise ValueError("No active experiment run. Call start_experiment() first.")
        
        # Log best configuration with prefix
        best_params = {f"best_{k}": v for k, v in best_config.items()}
        mlflow.log_params(best_params)
        
        # Log experiment summary metrics
        summary_metrics = {
            "final_best_score": best_score,
            "total_trials": total_trials
        }
        
        if experiment_duration:
            summary_metrics["experiment_duration_seconds"] = experiment_duration
        
        mlflow.log_metrics(summary_metrics)
        
        # Log experiment completion metadata
        end_time = datetime.now()
        mlflow.set_tags({
            "end_time": end_time.isoformat(),
            "experiment_duration_seconds": experiment_duration,
            "status": "completed"
        })
    
    def log_experiment_artifact(self, local_path: str, artifact_path: str = None):
        """
        Log an artifact to the parent experiment run.
        
        Args:
            local_path: Path to the local file to log
            artifact_path: Optional path within the run's artifact directory
        """
        if self.parent_run is None:
            raise ValueError("No active experiment run. Call start_experiment() first.")
        mlflow.log_artifact(local_path, artifact_path)
    
    def end_experiment(self):
        """End the main experiment run."""
        if self.parent_run:
            mlflow.end_run()
            self.parent_run = None
            self.experiment_start_time = None
    
    @contextmanager
    def experiment_run(self, run_name: str = None, experiment_config: Dict[str, Any] = None):
        """
        Context manager for experiment runs.
        
        Args:
            run_name: Optional name for the experiment run
            experiment_config: Configuration to log at experiment level
        """
        try:
            self.start_experiment(run_name, experiment_config)
            yield self
        finally:
            self.end_experiment()
    
    @contextmanager
    def trial_run(self, trial_id: str, trial_number: int, trial_config: Dict[str, Any] = None):
        """
        Context manager for trial runs.
        
        Args:
            trial_id: Unique identifier for the trial
            trial_number: Sequential trial number
            trial_config: Trial-specific configuration to log
        """
        try:
            self.start_trial(trial_id, trial_number, trial_config)
            yield self
        finally:
            self.end_trial()
    
    def get_experiment_info(self) -> Dict[str, Any]:
        """
        Get information about the current experiment.
        
        Returns:
            Dictionary with experiment information
        """
        if self.parent_run is None:
            return {}
        
        return {
            "experiment_name": self.experiment_name,
            "run_id": self.parent_run.info.run_id,
            "run_name": self.parent_run.info.run_name,
            "start_time": self.experiment_start_time.isoformat() if self.experiment_start_time else None
        }
    
    def get_trial_info(self) -> Dict[str, Any]:
        """
        Get information about the current trial.
        
        Returns:
            Dictionary with trial information
        """
        if self.current_trial_run is None:
            return {}
        
        return {
            "trial_run_id": self.current_trial_run.info.run_id,
            "trial_run_name": self.current_trial_run.info.run_name,
            "parent_run_id": self.parent_run.info.run_id if self.parent_run else None
        }