"""
Phase 5: Callback System for MLTeammate AutoML Controller

Modern, flexible callback system for experiment monitoring, logging, and artifact management.
Provides hooks into the AutoML experiment lifecycle with built-in MLflow integration.

This module provides:
- Abstract base callback interface
- MLflow experiment tracking
- Progress monitoring and logging
- Artifact management
- Performance visualization
- Custom callback extensibility
"""

import os
import json
import time
import pickle
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import our frozen Phase 1 utilities
from ml_teammate.utils import get_logger


class BaseCallback(ABC):
    """
    Abstract base class for AutoML experiment callbacks.
    
    Callbacks provide hooks into the AutoML experiment lifecycle:
    - on_experiment_start: Called when experiment begins
    - on_trial_start: Called before each trial
    - on_trial_end: Called after each trial
    - on_experiment_end: Called when experiment completes
    """
    
    def __init__(self, name: Optional[str] = None):
        """Initialize callback with optional name."""
        self.name = name or self.__class__.__name__
        self.logger = get_logger(f"Callback.{self.name}")
    
    def on_experiment_start(self, experiment_config: Dict[str, Any]):
        """Called when AutoML experiment starts."""
        pass
    
    def on_trial_start(self, trial_id: str, config: Dict[str, Any]):
        """Called before each trial starts."""
        pass
    
    def on_trial_end(self, trial_id: str, config: Dict[str, Any], 
                     score: float, is_best: bool):
        """Called after each trial completes."""
        pass
    
    def on_experiment_end(self, results: Dict[str, Any]):
        """Called when AutoML experiment completes."""
        pass


class LoggerCallback(BaseCallback):
    """
    Comprehensive logging callback with configurable verbosity.
    
    Provides detailed experiment logging with progress tracking,
    performance summaries, and configurable output formats.
    """
    
    def __init__(self, log_level: str = "INFO", log_trials: bool = True,
                 log_progress_interval: int = 5):
        """
        Initialize logger callback.
        
        Args:
            log_level: Logging level
            log_trials: Whether to log individual trials
            log_progress_interval: Log progress every N trials
        """
        super().__init__("Logger")
        self.log_trials = log_trials
        self.log_progress_interval = log_progress_interval
        self.logger = get_logger("AutoML.Logger", log_level)
        
        # State tracking
        self.start_time = None
        self.trial_count = 0
        self.best_score = None
        self.score_history = []
    
    def on_experiment_start(self, experiment_config: Dict[str, Any]):
        """Log experiment start with configuration."""
        self.start_time = time.time()
        self.trial_count = 0
        self.best_score = None
        self.score_history = []
        
        self.logger.info("=" * 60)
        self.logger.info("MLTeammate AutoML Experiment Started")
        self.logger.info("=" * 60)
        self.logger.info(f"Task: {experiment_config['task']}")
        self.logger.info(f"Learners: {experiment_config['learner_names']}")
        self.logger.info(f"Search: {experiment_config['searcher_type']}")
        self.logger.info(f"Trials: {experiment_config['n_trials']}")
        self.logger.info(f"Data: {experiment_config['data_shape']}")
        
        if experiment_config.get('cv_folds'):
            self.logger.info(f"Cross-validation: {experiment_config['cv_folds']} folds")
        
        self.logger.info("-" * 60)
    
    def on_trial_start(self, trial_id: str, config: Dict[str, Any]):
        """Log trial start."""
        if self.log_trials:
            learner = config.get('learner_name', 'unknown')
            self.logger.debug(f"Trial {self.trial_count + 1}: Starting {learner}")
    
    def on_trial_end(self, trial_id: str, config: Dict[str, Any], 
                     score: float, is_best: bool):
        """Log trial completion."""
        self.trial_count += 1
        self.score_history.append(score)
        
        if is_best:
            self.best_score = score
        
        # Log individual trial
        if self.log_trials:
            learner = config.get('learner_name', 'unknown')
            best_indicator = " ★ NEW BEST" if is_best else ""
            self.logger.info(f"Trial {self.trial_count}: {learner} -> {score:.4f}{best_indicator}")
        
        # Log progress summary
        if self.trial_count % self.log_progress_interval == 0:
            self._log_progress_summary()
    
    def on_experiment_end(self, results: Dict[str, Any]):
        """Log experiment completion summary."""
        duration = time.time() - self.start_time
        
        self.logger.info("-" * 60)
        self.logger.info("AutoML Experiment Completed")
        self.logger.info("-" * 60)
        
        exp_info = results.get("experiment_info", {})
        self.logger.info(f"Duration: {duration:.2f}s")
        self.logger.info(f"Trials completed: {self.trial_count}")
        self.logger.info(f"Success rate: {exp_info.get('success_rate', 0):.1%}")
        
        if results.get("best_score") is not None:
            self.logger.info(f"Best score: {results['best_score']:.4f}")
            self.logger.info(f"Best learner: {exp_info.get('best_learner', 'unknown')}")
            
            if exp_info.get('score_improvement'):
                improvement = exp_info['score_improvement']
                self.logger.info(f"Score improvement: +{improvement:.4f}")
        
        self.logger.info("=" * 60)
    
    def _log_progress_summary(self):
        """Log progress summary."""
        if not self.score_history:
            return
        
        recent_scores = self.score_history[-self.log_progress_interval:]
        avg_recent = np.mean(recent_scores)
        
        self.logger.info(f"Progress: {self.trial_count} trials, "
                        f"best={self.best_score:.4f}, "
                        f"recent_avg={avg_recent:.4f}")


class MLflowCallback(BaseCallback):
    """
    MLflow integration callback for experiment tracking.
    
    Automatically logs experiments, trials, metrics, parameters,
    and artifacts to MLflow for comprehensive experiment management.
    """
    
    def __init__(self, experiment_name: Optional[str] = None,
                 tracking_uri: Optional[str] = None,
                 log_artifacts: bool = True,
                 log_models: bool = True):
        """
        Initialize MLflow callback.
        
        Args:
            experiment_name: MLflow experiment name
            tracking_uri: MLflow tracking server URI
            log_artifacts: Whether to log artifacts
            log_models: Whether to log best model
        """
        super().__init__("MLflow")
        self.experiment_name = experiment_name or "MLTeammate_AutoML"
        self.log_artifacts = log_artifacts
        self.log_models = log_models
        
        # Import and configure MLflow
        try:
            import mlflow
            import mlflow.sklearn
            self.mlflow = mlflow
            self.mlflow_sklearn = mlflow.sklearn
            
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            
            # Set or create experiment
            try:
                self.experiment_id = mlflow.create_experiment(self.experiment_name)
            except Exception:
                # Experiment already exists
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                self.experiment_id = experiment.experiment_id
            
            self.logger.info(f"MLflow experiment: {self.experiment_name} ({self.experiment_id})")
            
        except ImportError:
            self.logger.warning("MLflow not available. Install with: pip install mlflow")
            self.mlflow = None
    
    def on_experiment_start(self, experiment_config: Dict[str, Any]):
        """Start MLflow run for experiment."""
        if not self.mlflow:
            return
        
        self.mlflow.start_run(experiment_id=self.experiment_id)
        
        # Log experiment parameters
        self.mlflow.log_params({
            "task": experiment_config["task"],
            "learner_names": ",".join(experiment_config["learner_names"]),
            "searcher_type": experiment_config["searcher_type"],
            "n_trials": experiment_config["n_trials"],
            "cv_folds": experiment_config.get("cv_folds"),
            "random_state": experiment_config["random_state"],
            "n_samples": experiment_config["n_samples"],
            "n_features": experiment_config["n_features"]
        })
        
        # Log additional tags
        self.mlflow.set_tags({
            "framework": "MLTeammate",
            "experiment_type": "automl",
            "data_shape": f"{experiment_config['n_samples']}x{experiment_config['n_features']}"
        })
    
    def on_trial_end(self, trial_id: str, config: Dict[str, Any], 
                     score: float, is_best: bool):
        """Log trial metrics to MLflow."""
        if not self.mlflow:
            return
        
        # Log trial score with step
        trial_idx = int(trial_id.split('_')[1])
        self.mlflow.log_metric("trial_score", score, step=trial_idx)
        
        # Log if this is the best trial
        if is_best:
            self.mlflow.log_metric("best_score", score)
    
    def on_experiment_end(self, results: Dict[str, Any]):
        """Log final results and artifacts to MLflow."""
        if not self.mlflow:
            return
        
        try:
            # Log final metrics
            if results.get("best_score") is not None:
                self.mlflow.log_metric("final_best_score", results["best_score"])
            
            exp_info = results.get("experiment_info", {})
            self.mlflow.log_metrics({
                "total_time": exp_info.get("total_time", 0),
                "completed_trials": exp_info.get("n_completed_trials", 0),
                "success_rate": exp_info.get("success_rate", 0),
                "score_improvement": exp_info.get("score_improvement", 0)
            })
            
            # Log best model
            if self.log_models and results.get("best_model") is not None:
                self.mlflow_sklearn.log_model(
                    results["best_model"], 
                    "best_model",
                    input_example=None  # Could add example input
                )
            
            # Log artifacts
            if self.log_artifacts:
                self._log_artifacts(results)
            
        finally:
            self.mlflow.end_run()
    
    def _log_artifacts(self, results: Dict[str, Any]):
        """Log experiment artifacts."""
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save results JSON
            results_file = temp_path / "results.json"
            with open(results_file, 'w') as f:
                # Make results JSON serializable
                serializable_results = self._make_json_serializable(results)
                json.dump(serializable_results, f, indent=2)
            self.mlflow.log_artifact(str(results_file))
            
            # Save search history plot
            if results.get("search_history"):
                plot_file = temp_path / "search_history.png"
                self._create_search_history_plot(results["search_history"], str(plot_file))
                self.mlflow.log_artifact(str(plot_file))
            
            # Save trial details
            if results.get("trials"):
                trials_file = temp_path / "trials.csv"
                trials_df = pd.DataFrame(results["trials"])
                trials_df.to_csv(trials_file, index=False)
                self.mlflow.log_artifact(str(trials_file))
    
    def _make_json_serializable(self, obj):
        """Convert object to JSON serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            return str(obj)  # Convert complex objects to string
        else:
            return obj
    
    def _create_search_history_plot(self, scores: List[float], filename: str):
        """Create and save search history plot."""
        plt.figure(figsize=(10, 6))
        plt.plot(scores, 'b-', alpha=0.7, label='Trial Scores')
        
        # Plot running best
        running_best = []
        best_so_far = scores[0] if scores else 0
        for score in scores:
            if score > best_so_far:
                best_so_far = score
            running_best.append(best_so_far)
        
        plt.plot(running_best, 'r-', linewidth=2, label='Best So Far')
        
        plt.xlabel('Trial')
        plt.ylabel('Score')
        plt.title('AutoML Search Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()


class ArtifactCallback(BaseCallback):
    """
    Artifact management callback for saving experiment outputs.
    
    Saves models, configurations, results, and visualizations
    to local filesystem for later analysis and reproducibility.
    """
    
    def __init__(self, output_dir: str = "mlteammate_artifacts",
                 save_models: bool = True, save_plots: bool = True,
                 save_configs: bool = True):
        """
        Initialize artifact callback.
        
        Args:
            output_dir: Directory to save artifacts
            save_models: Whether to save best model
            save_plots: Whether to save visualization plots
            save_configs: Whether to save trial configurations
        """
        super().__init__("Artifacts")
        self.output_dir = Path(output_dir)
        self.save_models = save_models
        self.save_plots = save_plots
        self.save_configs = save_configs
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Experiment-specific directory
        self.experiment_dir = None
    
    def on_experiment_start(self, experiment_config: Dict[str, Any]):
        """Create experiment directory and save config."""
        experiment_id = experiment_config.get("controller_id", "unknown")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        self.experiment_dir = self.output_dir / f"experiment_{timestamp}_{experiment_id[:8]}"
        self.experiment_dir.mkdir(exist_ok=True, parents=True)
        
        # Save experiment configuration
        if self.save_configs:
            config_file = self.experiment_dir / "experiment_config.json"
            with open(config_file, 'w') as f:
                json.dump(experiment_config, f, indent=2, default=str)
        
        self.logger.info(f"Artifacts will be saved to: {self.experiment_dir}")
    
    def on_experiment_end(self, results: Dict[str, Any]):
        """Save final experiment artifacts."""
        if not self.experiment_dir:
            return
        
        # Save complete results
        results_file = self.experiment_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save best model
        if self.save_models and results.get("best_model") is not None:
            model_file = self.experiment_dir / "best_model.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(results["best_model"], f)
        
        # Save trial configurations
        if self.save_configs and results.get("trials"):
            configs_file = self.experiment_dir / "trial_configs.json"
            trial_configs = [trial["config"] for trial in results["trials"]]
            with open(configs_file, 'w') as f:
                json.dump(trial_configs, f, indent=2, default=str)
        
        # Save plots
        if self.save_plots:
            self._save_plots(results)
        
        self.logger.info(f"Artifacts saved to: {self.experiment_dir}")
    
    def _save_plots(self, results: Dict[str, Any]):
        """Save visualization plots."""
        # Search history plot
        if results.get("search_history"):
            plt.figure(figsize=(12, 8))
            
            scores = results["search_history"]
            trials = range(1, len(scores) + 1)
            
            # Main score plot
            plt.subplot(2, 2, 1)
            plt.plot(trials, scores, 'b-', alpha=0.7, marker='o', markersize=3)
            plt.xlabel('Trial')
            plt.ylabel('Score')
            plt.title('Trial Scores')
            plt.grid(True, alpha=0.3)
            
            # Running best
            plt.subplot(2, 2, 2)
            running_best = []
            best_so_far = scores[0] if scores else 0
            for score in scores:
                if score > best_so_far:
                    best_so_far = score
                running_best.append(best_so_far)
            
            plt.plot(trials, running_best, 'r-', linewidth=2)
            plt.xlabel('Trial')
            plt.ylabel('Best Score')
            plt.title('Best Score Over Time')
            plt.grid(True, alpha=0.3)
            
            # Score distribution
            plt.subplot(2, 2, 3)
            plt.hist(scores, bins=min(20, len(scores)//2), alpha=0.7, edgecolor='black')
            plt.xlabel('Score')
            plt.ylabel('Frequency')
            plt.title('Score Distribution')
            plt.grid(True, alpha=0.3)
            
            # Improvement over trials
            plt.subplot(2, 2, 4)
            if len(scores) > 1:
                improvements = np.diff(running_best)
                plt.plot(trials[1:], improvements, 'g-', marker='o', markersize=3)
                plt.xlabel('Trial')
                plt.ylabel('Score Improvement')
                plt.title('Score Improvement per Trial')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.experiment_dir / "search_analysis.png", 
                       dpi=150, bbox_inches='tight')
            plt.close()


class ProgressCallback(BaseCallback):
    """
    Progress monitoring callback with live updates.
    
    Provides real-time progress tracking with ETA estimation,
    performance summaries, and configurable update intervals.
    """
    
    def __init__(self, update_interval: int = 5, show_eta: bool = True):
        """
        Initialize progress callback.
        
        Args:
            update_interval: Update progress every N trials
            show_eta: Whether to show estimated time to completion
        """
        super().__init__("Progress")
        self.update_interval = update_interval
        self.show_eta = show_eta
        
        # State tracking
        self.start_time = None
        self.total_trials = 0
        self.completed_trials = 0
        self.scores = []
    
    def on_experiment_start(self, experiment_config: Dict[str, Any]):
        """Initialize progress tracking."""
        self.start_time = time.time()
        self.total_trials = experiment_config["n_trials"]
        self.completed_trials = 0
        self.scores = []
        
        print(f"\n🚀 Starting AutoML with {self.total_trials} trials...")
        print("=" * 50)
    
    def on_trial_end(self, trial_id: str, config: Dict[str, Any], 
                     score: float, is_best: bool):
        """Update progress after each trial."""
        self.completed_trials += 1
        self.scores.append(score)
        
        if self.completed_trials % self.update_interval == 0:
            self._show_progress()
    
    def on_experiment_end(self, results: Dict[str, Any]):
        """Show final progress summary."""
        self._show_final_summary(results)
    
    def _show_progress(self):
        """Display current progress."""
        elapsed = time.time() - self.start_time
        progress_pct = (self.completed_trials / self.total_trials) * 100
        
        # Calculate ETA
        eta_str = ""
        if self.show_eta and self.completed_trials > 0:
            time_per_trial = elapsed / self.completed_trials
            remaining_trials = self.total_trials - self.completed_trials
            eta_seconds = time_per_trial * remaining_trials
            eta_str = f" | ETA: {eta_seconds:.0f}s"
        
        # Performance summary
        best_score = max(self.scores) if self.scores else 0
        avg_score = np.mean(self.scores) if self.scores else 0
        
        print(f"Progress: {self.completed_trials}/{self.total_trials} "
              f"({progress_pct:.1f}%) | "
              f"Best: {best_score:.4f} | "
              f"Avg: {avg_score:.4f}{eta_str}")
    
    def _show_final_summary(self, results: Dict[str, Any]):
        """Show final experiment summary."""
        elapsed = time.time() - self.start_time
        
        print("=" * 50)
        print("🎯 AutoML Completed!")
        print(f"⏱️  Total time: {elapsed:.2f}s")
        print(f"🧪 Trials: {self.completed_trials}/{self.total_trials}")
        
        if results.get("best_score") is not None:
            print(f"🏆 Best score: {results['best_score']:.4f}")
            
            exp_info = results.get("experiment_info", {})
            if exp_info.get("best_learner"):
                print(f"🤖 Best learner: {exp_info['best_learner']}")
        
        print("=" * 50)


def create_default_callbacks(log_level: str = "INFO", 
                           output_dir: str = "mlteammate_artifacts",
                           use_mlflow: bool = True,
                           mlflow_experiment: Optional[str] = None) -> List[BaseCallback]:
    """
    Create default set of callbacks for AutoML experiments.
    
    Args:
        log_level: Logging level
        output_dir: Directory for artifacts
        use_mlflow: Whether to include MLflow callback
        mlflow_experiment: MLflow experiment name
        
    Returns:
        List of configured callbacks
    """
    callbacks = [
        LoggerCallback(log_level=log_level),
        ProgressCallback(update_interval=5),
        ArtifactCallback(output_dir=output_dir)
    ]
    
    if use_mlflow:
        callbacks.append(MLflowCallback(experiment_name=mlflow_experiment))
    
    return callbacks
