"""
Phase 5: Tuner Interface for MLTeammate

Provides a unified interface for hyperparameter tuning that abstracts
different optimization libraries (Optuna, FLAML, etc.) behind a common API.

This module provides:
- AbstractTuner interface
- Library-specific tuner implementations
- Configuration validation and conversion
- Progress tracking and early stopping
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, Union
import time

from ml_teammate.utils import get_logger, ValidationError, time_context


class AbstractTuner(ABC):
    """
    Abstract base class for hyperparameter tuners.
    
    Provides a unified interface for different optimization libraries
    while allowing library-specific customizations.
    """
    
    def __init__(self, objective_name: str = "score", direction: str = "maximize",
                 random_state: int = 42, log_level: str = "INFO"):
        """
        Initialize tuner.
        
        Args:
            objective_name: Name of the objective to optimize
            direction: Optimization direction ("maximize" or "minimize")
            random_state: Random seed for reproducibility
            log_level: Logging level
        """
        self.objective_name = objective_name
        self.direction = direction
        self.random_state = random_state
        self.logger = get_logger(f"Tuner.{self.__class__.__name__}", log_level)
        
        # Validate direction
        if direction not in ["maximize", "minimize"]:
            raise ValidationError(f"Invalid direction: {direction}")
        
        # Initialize tracking
        self.n_trials = 0
        self.best_value = None
        self.best_params = None
        self.study_start_time = None
        
        self.logger.info(f"Tuner initialized: {direction} {objective_name}")
    
    @abstractmethod
    def suggest_trial(self, trial_number: int) -> Dict[str, Any]:
        """
        Suggest hyperparameters for the next trial.
        
        Args:
            trial_number: Current trial number
            
        Returns:
            Dictionary of suggested hyperparameters
        """
        pass
    
    @abstractmethod
    def report_trial(self, trial_number: int, value: float, 
                    params: Dict[str, Any], additional_info: Optional[Dict] = None):
        """
        Report the result of a trial.
        
        Args:
            trial_number: Trial number
            value: Objective value to optimize
            params: Parameters that were evaluated
            additional_info: Optional additional trial information
        """
        pass
    
    @abstractmethod
    def get_best_trial(self) -> Dict[str, Any]:
        """
        Get the best trial found so far.
        
        Returns:
            Dictionary with best trial information
        """
        pass
    
    def start_study(self):
        """Start the optimization study."""
        self.study_start_time = time.time()
        self.logger.info("Optimization study started")
    
    def finish_study(self) -> Dict[str, Any]:
        """
        Finish the study and return summary.
        
        Returns:
            Study summary with best results and statistics
        """
        if self.study_start_time is None:
            study_duration = 0.0
        else:
            study_duration = time.time() - self.study_start_time
        
        summary = {
            "best_value": self.best_value,
            "best_params": self.best_params,
            "n_trials": self.n_trials,
            "study_duration": study_duration,
            "objective_name": self.objective_name,
            "direction": self.direction
        }
        
        self.logger.info(f"Study completed: {self.n_trials} trials, "
                        f"best {self.objective_name} = {self.best_value:.4f}")
        
        return summary
    
    def _update_best(self, value: float, params: Dict[str, Any]):
        """Update best trial if current is better."""
        is_better = (
            self.best_value is None or
            (self.direction == "maximize" and value > self.best_value) or
            (self.direction == "minimize" and value < self.best_value)
        )
        
        if is_better:
            self.best_value = value
            self.best_params = params.copy()
            return True
        
        return False


class SimpleTuner(AbstractTuner):
    """
    Simple tuner that works without external optimization libraries.
    
    Uses basic optimization strategies like random search or grid search
    as a fallback when advanced libraries are not available.
    """
    
    def __init__(self, param_space: Dict[str, Any], strategy: str = "random",
                 **kwargs):
        """
        Initialize simple tuner.
        
        Args:
            param_space: Parameter search space definition
            strategy: Tuning strategy ("random", "grid")
            **kwargs: Additional arguments for AbstractTuner
        """
        super().__init__(**kwargs)
        self.param_space = param_space
        self.strategy = strategy
        self.trial_history = []
        
        if strategy not in ["random", "grid"]:
            raise ValidationError(f"Invalid strategy: {strategy}")
        
        self.logger.info(f"SimpleTuner initialized with {strategy} strategy")
    
    def suggest_trial(self, trial_number: int) -> Dict[str, Any]:
        """Suggest parameters using simple strategy."""
        with time_context("SimpleTuner.suggest_trial"):
            if self.strategy == "random":
                return self._random_suggest()
            elif self.strategy == "grid":
                return self._grid_suggest(trial_number)
            else:
                raise ValidationError(f"Unknown strategy: {self.strategy}")
    
    def _random_suggest(self) -> Dict[str, Any]:
        """Random parameter suggestion."""
        import random
        import numpy as np
        
        # Set random seed for reproducibility
        random.seed(self.random_state + self.n_trials)
        np.random.seed(self.random_state + self.n_trials)
        
        params = {}
        for param_name, param_spec in self.param_space.items():
            param_type = param_spec.get("type", "float")
            
            if param_type == "int":
                low, high = param_spec["bounds"]
                params[param_name] = random.randint(low, high)
            
            elif param_type == "float":
                low, high = param_spec["bounds"]
                log_scale = param_spec.get("log_scale", False)
                
                if log_scale:
                    log_low, log_high = np.log(low), np.log(high)
                    log_value = random.uniform(log_low, log_high)
                    params[param_name] = np.exp(log_value)
                else:
                    params[param_name] = random.uniform(low, high)
            
            elif param_type == "categorical":
                choices = param_spec["choices"]
                params[param_name] = random.choice(choices)
            
            elif param_type == "fixed":
                params[param_name] = param_spec["value"]
        
        return params
    
    def _grid_suggest(self, trial_number: int) -> Dict[str, Any]:
        """Grid search parameter suggestion."""
        # For simplicity, fall back to random for now
        # A full grid search implementation would require
        # pre-computing all combinations
        return self._random_suggest()
    
    def report_trial(self, trial_number: int, value: float, 
                    params: Dict[str, Any], additional_info: Optional[Dict] = None):
        """Report trial result."""
        # Update tracking
        self.n_trials += 1
        
        # Store trial info
        trial_info = {
            "trial_number": trial_number,
            "value": value,
            "params": params.copy(),
            "additional_info": additional_info or {}
        }
        self.trial_history.append(trial_info)
        
        # Update best
        is_best = self._update_best(value, params)
        
        self.logger.debug(f"Trial {trial_number}: {self.objective_name} = {value:.4f} "
                         f"({'BEST' if is_best else 'ok'})")
    
    def get_best_trial(self) -> Dict[str, Any]:
        """Get best trial information."""
        if self.best_value is None:
            return {"value": None, "params": None}
        
        return {
            "value": self.best_value,
            "params": self.best_params,
            "trial_number": len(self.trial_history)
        }


def create_tuner(tuner_type: str, param_space: Dict[str, Any], **kwargs) -> AbstractTuner:
    """
    Factory function to create tuner instances.
    
    Args:
        tuner_type: Type of tuner ("simple", "optuna", "skopt")
        param_space: Parameter search space
        **kwargs: Additional tuner arguments
        
    Returns:
        AbstractTuner: Configured tuner instance
    """
    if tuner_type == "simple":
        return SimpleTuner(param_space, **kwargs)
    
    # Try to create advanced tuners if libraries are available
    elif tuner_type == "optuna":
        try:
            from ..search.optuna_search import OptunaSearcher
            return SimpleTuner(param_space, strategy="optuna", **kwargs)
        except ImportError:
            logger = get_logger("TunerFactory")
            logger.warning("Optuna not available, falling back to SimpleTuner")
            return SimpleTuner(param_space, strategy="random", **kwargs)
    
    elif tuner_type == "skopt":
        logger = get_logger("TunerFactory")
        logger.warning("Scikit-optimize tuner not implemented, falling back to SimpleTuner")
        return SimpleTuner(param_space, strategy="random", **kwargs)
    
    else:
        available_tuners = ["simple", "optuna", "skopt"]
        raise ValidationError(f"Unknown tuner type: {tuner_type}. "
                            f"Available: {available_tuners}")


def get_available_tuners() -> List[str]:
    """
    Get list of available tuner types.
    
    Returns:
        List of tuner type names
    """
    tuners = ["simple"]
    
    # Check for optional dependencies
    try:
        import optuna
        tuners.append("optuna")
    except ImportError:
        pass
    
    # Note: skopt is not currently supported
    # tuners.append("skopt") would go here if implemented
    
    return tuners
