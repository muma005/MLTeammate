# ml_teammate/automl/callbacks.py
"""
Enhanced callback system for MLTeammate.
Provides flexible logging, monitoring, and artifact management capabilities.
"""

import time
import json
import os
from typing import Dict, Any, Optional
from datetime import datetime

class BaseCallback:
    """Base class for all callbacks in MLTeammate."""
    
    def on_trial_start(self, trial_id: str, config: dict) -> None:
        """Called when a trial starts."""
        pass
    
    def on_trial_end(self, trial_id: str, config: dict, score: float, is_best: bool) -> None:
        """Called when a trial ends."""
        pass
    
    def on_experiment_start(self, experiment_config: dict) -> None:
        """Called when the experiment starts."""
        pass
    
    def on_experiment_end(self, best_score: float, best_config: dict) -> None:
        """Called when the experiment ends."""
        pass


class LoggerCallback(BaseCallback):
    """
    Enhanced logging callback with structured output and optional MLflow integration.
    
    Features:
    - Structured logging with timestamps
    - Configurable log levels
    - Optional MLflow integration
    - Trial progress tracking
    - Best model highlighting
    """
    
    def __init__(self, 
                 use_mlflow: bool = False,
                 log_level: str = "INFO",
                 log_file: Optional[str] = None,
                 experiment_name: str = "mlteammate_experiment"):
        """
        Initialize the logger callback.
        
        Args:
            use_mlflow: Whether to enable MLflow logging
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            log_file: Optional file path for logging
            experiment_name: Name for MLflow experiment
        """
        self.use_mlflow = use_mlflow
        self.log_level = log_level.upper()
        self.log_file = log_file
        self.experiment_name = experiment_name
        self.trial_count = 0
        self.start_time = None
        
        # Initialize MLflow if requested
        if use_mlflow:
            try:
                from ml_teammate.experiments.mlflow_helper import MLflowHelper
                self.mlflow = MLflowHelper(experiment_name)
            except ImportError:
                print("⚠️ MLflow not available. Continuing without MLflow logging.")
                self.use_mlflow = False
                self.mlflow = None
        else:
            self.mlflow = None
    
    def _log(self, message: str, level: str = "INFO") -> None:
        """Internal logging method with level filtering."""
        if self._should_log(level):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            formatted_message = f"[{timestamp}] [{level}] {message}"
            print(formatted_message)
            
            if self.log_file:
                with open(self.log_file, 'a') as f:
                    f.write(formatted_message + '\n')
    
    def _should_log(self, level: str) -> bool:
        """Check if message should be logged based on log level."""
        levels = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
        return levels.get(level, 1) >= levels.get(self.log_level, 1)
    
    def on_experiment_start(self, experiment_config: dict) -> None:
        """Log experiment start with configuration."""
        self.start_time = time.time()
        self._log(f"🚀 Starting MLTeammate experiment: {self.experiment_name}")
        self._log(f"📋 Experiment config: {json.dumps(experiment_config, indent=2)}")
        
        if self.use_mlflow and self.mlflow:
            try:
                self.mlflow.start_run(run_name=f"{self.experiment_name}_{int(time.time())}")
                self._log("✅ MLflow run started")
            except Exception as e:
                self._log(f"❌ Failed to start MLflow run: {e}", "ERROR")
    
    def on_trial_start(self, trial_id: str, config: dict) -> None:
        """Log trial start with configuration."""
        self.trial_count += 1
        self._log(f"🔬 Starting trial {self.trial_count} (ID: {trial_id[:8]}...)")
        self._log(f"⚙️  Config: {json.dumps(config, indent=2)}")
    
    def on_trial_end(self, trial_id: str, config: dict, score: float, is_best: bool) -> None:
        """Log trial results with enhanced formatting."""
        duration = time.time() - self.start_time if self.start_time else 0
        
        # Format the log message
        status_icon = "🏆" if is_best else "📊"
        best_indicator = " (NEW BEST!)" if is_best else ""
        
        self._log(f"{status_icon} Trial {self.trial_count} completed - Score: {score:.4f}{best_indicator}")
        self._log(f"⏱️  Duration: {duration:.2f}s")
        
        # Log to MLflow if enabled
        if self.use_mlflow and self.mlflow:
            try:
                self.mlflow.log_params(config)
                self.mlflow.log_metrics({"score": score, "trial_number": self.trial_count})
                if is_best:
                    self.mlflow.log_metrics({"best_score": score})
            except Exception as e:
                self._log(f"❌ MLflow logging failed: {e}", "ERROR")
    
    def on_experiment_end(self, best_score: float, best_config: dict) -> None:
        """Log experiment completion with summary."""
        total_duration = time.time() - self.start_time if self.start_time else 0
        
        self._log("🎉 Experiment completed!")
        self._log(f"📈 Best score: {best_score:.4f}")
        self._log(f"⚙️  Best config: {json.dumps(best_config, indent=2)}")
        self._log(f"⏱️  Total duration: {total_duration:.2f}s")
        self._log(f"🔬 Total trials: {self.trial_count}")
        
        if self.use_mlflow and self.mlflow:
            try:
                self.mlflow.log_metrics({"final_best_score": best_score, "total_trials": self.trial_count})
                self.mlflow.end_run()
                self._log("✅ MLflow run completed")
            except Exception as e:
                self._log(f"❌ Failed to end MLflow run: {e}", "ERROR")


class ProgressCallback(BaseCallback):
    """
    Progress tracking callback for real-time experiment monitoring.
    
    Features:
    - Progress bars and percentage completion
    - ETA calculations
    - Performance trend analysis
    - Early stopping suggestions
    """
    
    def __init__(self, total_trials: int, patience: int = 5):
        """
        Initialize progress callback.
        
        Args:
            total_trials: Total number of trials to run
            patience: Number of trials without improvement before suggesting early stopping
        """
        self.total_trials = total_trials
        self.patience = patience
        self.completed_trials = 0
        self.best_score = None
        self.trials_since_improvement = 0
        self.start_time = None
    
    def on_experiment_start(self, experiment_config: dict) -> None:
        """Initialize progress tracking."""
        self.start_time = time.time()
        print(f"🎯 Progress tracking enabled for {self.total_trials} trials")
    
    def on_trial_end(self, trial_id: str, config: dict, score: float, is_best: bool) -> None:
        """Update progress and show status."""
        self.completed_trials += 1
        
        if is_best:
            self.best_score = score
            self.trials_since_improvement = 0
        else:
            self.trials_since_improvement += 1
        
        # Calculate progress
        progress = (self.completed_trials / self.total_trials) * 100
        eta = self._calculate_eta()
        
        # Show progress bar
        bar_length = 30
        filled_length = int(bar_length * self.completed_trials // self.total_trials)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        print(f"\r📊 Progress: [{bar}] {progress:.1f}% ({self.completed_trials}/{self.total_trials}) "
              f"| Best: {self.best_score:.4f} | ETA: {eta}", end='')
        
        # Suggest early stopping if no improvement
        if self.trials_since_improvement >= self.patience:
            print(f"\n⚠️  No improvement for {self.patience} trials. Consider early stopping.")
    
    def _calculate_eta(self) -> str:
        """Calculate estimated time to completion."""
        if self.completed_trials == 0:
            return "Unknown"
        
        elapsed = time.time() - self.start_time
        avg_time_per_trial = elapsed / self.completed_trials
        remaining_trials = self.total_trials - self.completed_trials
        eta_seconds = avg_time_per_trial * remaining_trials
        
        if eta_seconds < 60:
            return f"{eta_seconds:.0f}s"
        elif eta_seconds < 3600:
            return f"{eta_seconds/60:.1f}m"
        else:
            return f"{eta_seconds/3600:.1f}h"
    
    def on_experiment_end(self, best_score: float, best_config: dict) -> None:
        """Show final progress summary."""
        total_duration = time.time() - self.start_time
        print(f"\n✅ Experiment completed in {total_duration:.2f}s")


class ArtifactCallback(BaseCallback):
    """
    Artifact management callback for saving models, plots, and other outputs.
    
    Features:
    - Automatic model saving
    - Feature importance plots
    - Configuration dumps
    - Performance visualizations
    """
    
    def __init__(self, 
                 save_best_model: bool = True,
                 save_configs: bool = True,
                 output_dir: str = "./mlteammate_artifacts"):
        """
        Initialize artifact callback.
        
        Args:
            save_best_model: Whether to save the best model
            save_configs: Whether to save trial configurations
            output_dir: Directory to save artifacts
        """
        self.save_best_model = save_best_model
        self.save_configs = save_configs
        self.output_dir = output_dir
        self.best_model = None
        self.best_config = None
        self.all_configs = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def on_trial_end(self, trial_id: str, config: dict, score: float, is_best: bool) -> None:
        """Store trial information for artifact generation."""
        trial_info = {
            "trial_id": trial_id,
            "config": config,
            "score": score,
            "is_best": is_best,
            "timestamp": datetime.now().isoformat()
        }
        self.all_configs.append(trial_info)
    
    def on_experiment_end(self, best_score: float, best_config: dict) -> None:
        """Generate and save artifacts."""
        self.best_config = best_config
        
        if self.save_configs:
            self._save_configs()
        
        print(f"📁 Artifacts saved to: {self.output_dir}")
    
    def _save_configs(self) -> None:
        """Save all trial configurations to JSON file."""
        config_file = os.path.join(self.output_dir, "trial_configs.json")
        with open(config_file, 'w') as f:
            json.dump(self.all_configs, f, indent=2)
    
    def save_model(self, model, filename: str = "best_model.pkl") -> None:
        """Save the best model to disk."""
        if not self.save_best_model:
            return
        
        try:
            import joblib
            model_path = os.path.join(self.output_dir, filename)
            joblib.dump(model, model_path)
            print(f"💾 Model saved to: {model_path}")
        except ImportError:
            print("⚠️ joblib not available. Skipping model save.")
        except Exception as e:
            print(f"❌ Failed to save model: {e}")


# Factory function for easy callback creation
def create_callbacks(logging: bool = True, 
                    progress: bool = True, 
                    artifacts: bool = True,
                    **kwargs) -> list:
    """
    Create a list of callbacks with common configurations.
    
    Args:
        logging: Whether to include LoggerCallback
        progress: Whether to include ProgressCallback
        artifacts: Whether to include ArtifactCallback
        **kwargs: Additional arguments for callbacks
    
    Returns:
        List of configured callbacks
    """
    callbacks = []
    
    if logging:
        callbacks.append(LoggerCallback(**kwargs.get('logging', {})))
    
    if progress and 'total_trials' in kwargs:
        callbacks.append(ProgressCallback(**kwargs.get('progress', {})))
    
    if artifacts:
        callbacks.append(ArtifactCallback(**kwargs.get('artifacts', {})))
    
    return callbacks
