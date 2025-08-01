"""
FLAML Searcher for MLTeammate

This module provides a FLAML-based hyperparameter optimization searcher
that integrates with MLTeammate's search interface.

FLAML (Fast and Lightweight AutoML) provides efficient hyperparameter
optimization with built-in early stopping and resource management.
"""

import time
import uuid
from typing import Dict, Any, Optional, List
import numpy as np

try:
    import flaml
    from flaml import AutoML
    FLAML_AVAILABLE = True
except ImportError:
    FLAML_AVAILABLE = False
    print("⚠️ FLAML not available. Install with: pip install flaml")


class FLAMLSearcher:
    """
    FLAML-based hyperparameter optimization searcher.
    
    FLAML provides efficient hyperparameter optimization with:
    - Built-in early stopping
    - Resource management
    - Multiple optimization algorithms
    - Time budget management
    """
    
    def __init__(self, 
                 config_spaces: Dict[str, Dict],
                 time_budget: int = 60,
                 metric: str = "accuracy",
                 mode: str = "max",
                 estimator_list: Optional[List[str]] = None,
                 **flaml_kwargs):
        """
        Initialize FLAML searcher.
        
        Args:
            config_spaces: Configuration spaces for each learner
            time_budget: Time budget in seconds for optimization
            metric: Metric to optimize (e.g., "accuracy", "mse", "mae")
            mode: Optimization mode ("max" or "min")
            estimator_list: List of estimators to try (e.g., ["lgbm", "xgboost", "rf"])
            **flaml_kwargs: Additional FLAML parameters
        """
        if not FLAML_AVAILABLE:
            raise ImportError("FLAML is not available. Install with: pip install flaml")
        
        self.config_spaces = config_spaces
        self.time_budget = time_budget
        self.metric = metric
        self.mode = mode
        self.estimator_list = estimator_list or ["lgbm", "xgboost", "rf"]
        self.flaml_kwargs = flaml_kwargs
        
        # FLAML AutoML instance
        self.automl = AutoML()
        
        # Trial tracking
        self.trials = {}
        self.best_score = None
        self.best_config = None
        self.start_time = None
        
        # FLAML settings
        self.settings = {
            "time_budget": time_budget,
            "metric": metric,
            "mode": mode,
            "estimator_list": self.estimator_list,
            "task": "classification",  # Will be set dynamically
            "log_file_name": None,
            "verbose": 0,
            **flaml_kwargs
        }
    
    def _convert_config_space(self, learner_name: str) -> Dict[str, Any]:
        """
        Convert MLTeammate config space to FLAML format.
        
        Args:
            learner_name: Name of the learner
            
        Returns:
            FLAML-compatible configuration space
        """
        space = self.config_spaces.get(learner_name, {})
        flaml_space = {}
        
        for param_name, param_spec in space.items():
            param_type = param_spec["type"]
            
            if param_type == "int":
                low, high = param_spec["bounds"]
                flaml_space[param_name] = {
                    "domain": "uniform",
                    "init_value": (low + high) // 2,
                    "low": low,
                    "high": high
                }
            elif param_type == "float":
                low, high = param_spec["bounds"]
                flaml_space[param_name] = {
                    "domain": "uniform",
                    "init_value": (low + high) / 2,
                    "low": low,
                    "high": high
                }
            elif param_type == "categorical":
                choices = param_spec["choices"]
                flaml_space[param_name] = {
                    "domain": "choice",
                    "init_value": choices[0],
                    "choices": choices
                }
        
        return flaml_space
    
    def suggest(self, trial_id: str, learner_name: str) -> Dict[str, Any]:
        """
        Suggest hyperparameters for a trial.
        
        Note: FLAML works differently from Optuna - it optimizes
        the entire search process rather than suggesting individual
        configurations. This method returns a placeholder config
        that will be overridden by FLAML's internal optimization.
        
        Args:
            trial_id: Unique trial identifier
            learner_name: Name of the learner
            
        Returns:
            Suggested configuration (placeholder for FLAML)
        """
        # For FLAML, we return a placeholder config
        # The actual optimization happens in the fit method
        space = self.config_spaces.get(learner_name, {})
        config = {}
        
        for param_name, param_spec in space.items():
            param_type = param_spec["type"]
            
            if param_type == "int":
                low, high = param_spec["bounds"]
                config[param_name] = (low + high) // 2
            elif param_type == "float":
                low, high = param_spec["bounds"]
                config[param_name] = (low + high) / 2
            elif param_type == "categorical":
                choices = param_spec["choices"]
                config[param_name] = choices[0]
        
        # Store trial info
        self.trials[trial_id] = {
            "learner_name": learner_name,
            "config": config,
            "start_time": time.time()
        }
        
        return config
    
    def report(self, trial_id: str, score: float):
        """
        Report the score for a trial.
        
        Note: In FLAML, this is mainly for tracking purposes
        as FLAML handles optimization internally.
        
        Args:
            trial_id: Trial identifier
            score: Trial score
        """
        if trial_id in self.trials:
            self.trials[trial_id]["score"] = score
            self.trials[trial_id]["end_time"] = time.time()
            
            # Update best score
            if self.best_score is None:
                self.best_score = score
                self.best_config = self.trials[trial_id]["config"]
            elif self.mode == "max" and score > self.best_score:
                self.best_score = score
                self.best_config = self.trials[trial_id]["config"]
            elif self.mode == "min" and score < self.best_score:
                self.best_score = score
                self.best_config = self.trials[trial_id]["config"]
    
    def fit(self, X, y, task: str = "classification"):
        """
        Run FLAML optimization.
        
        This is the main method for FLAML optimization. Unlike Optuna,
        FLAML optimizes the entire process rather than individual trials.
        
        Args:
            X: Training features
            y: Training targets
            task: Task type ("classification" or "regression")
            
        Returns:
            Best configuration found by FLAML
        """
        if not FLAML_AVAILABLE:
            raise ImportError("FLAML is not available")
        
        self.start_time = time.time()
        
        # Update settings for the task
        self.settings["task"] = task
        
        # Run FLAML optimization
        self.automl.fit(X, y, **self.settings)
        
        # Extract best configuration
        best_config = self.automl.best_config
        self.best_score = self.automl.best_loss if self.mode == "min" else -self.automl.best_loss
        self.best_config = best_config
        
        return best_config
    
    def get_best(self) -> Dict[str, Any]:
        """
        Get the best configuration found.
        
        Returns:
            Best configuration dictionary
        """
        if self.best_config is None:
            return {}
        
        # Convert FLAML config to standard format
        result = {
            "learner_name": "flaml_optimized",
            "score": self.best_score,
            "config": self.best_config.copy()
        }
        
        # Add FLAML-specific information
        if hasattr(self.automl, 'best_estimator'):
            result["estimator"] = type(self.automl.best_estimator).__name__
        
        return result
    
    def get_trial_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of all trials.
        
        Returns:
            List of trial information dictionaries
        """
        history = []
        for trial_id, trial_info in self.trials.items():
            history.append({
                "trial_id": trial_id,
                "learner_name": trial_info.get("learner_name"),
                "config": trial_info.get("config", {}),
                "score": trial_info.get("score"),
                "duration": trial_info.get("end_time", 0) - trial_info.get("start_time", 0)
            })
        return history
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the optimization process.
        
        Returns:
            Optimization summary dictionary
        """
        if self.start_time is None:
            return {}
        
        total_time = time.time() - self.start_time
        n_trials = len(self.trials)
        
        return {
            "total_time": total_time,
            "n_trials": n_trials,
            "best_score": self.best_score,
            "time_budget": self.time_budget,
            "metric": self.metric,
            "mode": self.mode,
            "estimator_list": self.estimator_list,
            "flaml_version": flaml.__version__ if FLAML_AVAILABLE else None
        }


class FLAMLTimeBudgetSearcher(FLAMLSearcher):
    """
    FLAML searcher with time budget management.
    
    This variant focuses on time-bounded optimization,
    which is one of FLAML's key strengths.
    """
    
    def __init__(self, 
                 config_spaces: Dict[str, Dict],
                 time_budget: int = 60,
                 **kwargs):
        """
        Initialize FLAML time budget searcher.
        
        Args:
            config_spaces: Configuration spaces for each learner
            time_budget: Time budget in seconds
            **kwargs: Additional arguments for FLAMLSearcher
        """
        super().__init__(
            config_spaces=config_spaces,
            time_budget=time_budget,
            **kwargs
        )
        
        # FLAML-specific time budget settings
        self.settings.update({
            "time_budget": time_budget,
            "max_iter": None,  # Let FLAML handle iterations based on time
            "eval_method": "holdout",  # Use holdout for speed
            "split_ratio": 0.2,  # 20% for validation
        })


class FLAMLResourceAwareSearcher(FLAMLSearcher):
    """
    FLAML searcher with resource-aware optimization.
    
    This variant includes memory and computational resource
    considerations in the optimization process.
    """
    
    def __init__(self, 
                 config_spaces: Dict[str, Dict],
                 time_budget: int = 60,
                 memory_budget: Optional[int] = None,
                 **kwargs):
        """
        Initialize FLAML resource-aware searcher.
        
        Args:
            config_spaces: Configuration spaces for each learner
            time_budget: Time budget in seconds
            memory_budget: Memory budget in MB (optional)
            **kwargs: Additional arguments for FLAMLSearcher
        """
        super().__init__(
            config_spaces=config_spaces,
            time_budget=time_budget,
            **kwargs
        )
        
        # Resource-aware settings
        if memory_budget:
            self.settings["memory_budget"] = memory_budget
        
        # Use more conservative settings for resource-constrained environments
        self.settings.update({
            "estimator_list": ["lgbm", "rf"],  # Lighter estimators
            "eval_method": "holdout",
            "split_ratio": 0.2,
            "n_jobs": 1,  # Single-threaded to save resources
        })
