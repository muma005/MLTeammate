"""
Phase 4: Search System for MLTeammate

Clean, modern hyperparameter optimization system that integrates seamlessly
with the frozen Phase 3 learner registry and Phase 1 utilities.

This module provides:
- Optuna integration with all major samplers
- FLAML time-budget optimization
- Validated configuration spaces
- Early convergence detection
- Factory pattern for easy searcher creation
"""

import numpy as np
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
from abc import ABC, abstractmethod
import time

# Import our frozen Phase 1 utilities
from ml_teammate.utils import (
    get_logger, ValidationError, Timer, time_context
)

# Import our frozen Phase 3 learner registry
from ml_teammate.learners.registry import (
    get_learner_registry, create_learner, get_learner_config_space
)


class BaseSearcher(ABC):
    """
    Abstract base class for all hyperparameter searchers.
    
    Provides consistent interface and integration with frozen MLTeammate components.
    """
    
    def __init__(self, learner_names: List[str], task: str = "classification", 
                 random_state: int = 42, log_level: str = "INFO"):
        """
        Initialize base searcher.
        
        Args:
            learner_names: List of learner names to search over
            task: Task type ("classification" or "regression")
            random_state: Random seed for reproducibility
            log_level: Logging level
        """
        self.learner_names = learner_names
        self.task = task
        self.random_state = random_state
        self.logger = get_logger(f"Searcher_{self.__class__.__name__}", log_level)
        
        # Validate inputs
        if not learner_names:
            raise ValidationError("learner_names cannot be empty")
        
        if task not in ["classification", "regression"]:
            raise ValidationError(f"Invalid task: {task}")
        
        # Get registry and validate learners
        self.registry = get_learner_registry()
        self._validate_learners()
        
        # Build configuration spaces
        self.config_spaces = self._build_config_spaces()
        
        # Search state
        self.trials = {}
        self.best_score = None
        self.best_config = None
        self.n_trials = 0
        
        self.logger.info(f"Searcher initialized for {len(learner_names)} learners, task: {task}")
    
    def _validate_learners(self):
        """Validate that all learners exist and support the task."""
        available_learners = self.registry.list_learners(self.task)
        
        for learner_name in self.learner_names:
            if learner_name not in available_learners:
                available = list(available_learners.keys())
                raise ValidationError(f"Learner '{learner_name}' not available for {self.task}. "
                                    f"Available: {available}")
        
        self.logger.debug(f"Validated {len(self.learner_names)} learners for {self.task}")
    
    def _build_config_spaces(self) -> Dict[str, Dict[str, Any]]:
        """Build configuration spaces for all learners."""
        config_spaces = {}
        
        for learner_name in self.learner_names:
            try:
                config_space = get_learner_config_space(learner_name)
                config_spaces[learner_name] = config_space
                self.logger.debug(f"Config space for {learner_name}: {len(config_space)} parameters")
            except Exception as e:
                self.logger.warning(f"Failed to get config space for {learner_name}: {e}")
                config_spaces[learner_name] = {}
        
        return config_spaces
    
    @abstractmethod
    def suggest(self, trial_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Suggest next configuration to try.
        
        Args:
            trial_id: Optional trial identifier
            
        Returns:
            dict: Configuration with 'learner_name' and hyperparameters
        """
        pass
    
    @abstractmethod
    def report(self, trial_id: str, score: float, config: Dict[str, Any]) -> None:
        """
        Report trial result.
        
        Args:
            trial_id: Trial identifier
            score: Performance score (higher is better)
            config: Configuration that was evaluated
        """
        pass
    
    def create_learner_from_config(self, config: Dict[str, Any]):
        """
        Create learner instance from configuration.
        
        Args:
            config: Configuration dict with 'learner_name' and hyperparameters
            
        Returns:
            SklearnWrapper: Configured learner instance
        """
        if "learner_name" not in config:
            raise ValidationError("Configuration must contain 'learner_name'")
        
        learner_name = config["learner_name"]
        hyperparams = {k: v for k, v in config.items() if k != "learner_name"}
        
        return create_learner(learner_name, hyperparams)
    
    def get_search_summary(self) -> Dict[str, Any]:
        """Get comprehensive search summary."""
        return {
            "searcher_type": self.__class__.__name__,
            "task": self.task,
            "learner_names": self.learner_names,
            "n_trials": self.n_trials,
            "best_score": self.best_score,
            "best_config": self.best_config,
            "config_spaces": {name: len(space) for name, space in self.config_spaces.items()}
        }


class SearchResult:
    """
    Container for search results with comprehensive information.
    """
    
    def __init__(self, best_config: Dict[str, Any], best_score: float, 
                 all_trials: List[Dict[str, Any]], search_time: float,
                 searcher_info: Dict[str, Any]):
        """
        Initialize search result.
        
        Args:
            best_config: Best configuration found
            best_score: Best score achieved
            all_trials: List of all trial results
            search_time: Total search time in seconds
            searcher_info: Information about the searcher used
        """
        self.best_config = best_config
        self.best_score = best_score
        self.all_trials = all_trials
        self.search_time = search_time
        self.searcher_info = searcher_info
        
        # Derived statistics
        self.n_trials = len(all_trials)
        self.scores = [trial.get("score", 0.0) for trial in all_trials]
        
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive result summary."""
        return {
            "best_score": self.best_score,
            "best_config": self.best_config,
            "n_trials": self.n_trials,
            "search_time": self.search_time,
            "mean_score": np.mean(self.scores) if self.scores else 0.0,
            "std_score": np.std(self.scores) if self.scores else 0.0,
            "score_improvement": (self.best_score - self.scores[0]) if self.scores else 0.0,
            "searcher_info": self.searcher_info
        }
    
    def get_trial_history(self) -> List[float]:
        """Get score history for plotting."""
        return self.scores
    
    def create_best_learner(self):
        """Create learner instance with best configuration."""
        if not self.best_config:
            raise ValidationError("No best configuration available")
        
        learner_name = self.best_config["learner_name"]
        hyperparams = {k: v for k, v in self.best_config.items() if k != "learner_name"}
        
        return create_learner(learner_name, hyperparams)


def validate_config_against_space(config: Dict[str, Any], config_space: Dict[str, Any]) -> bool:
    """
    Validate configuration against configuration space definition.
    
    Args:
        config: Configuration to validate
        config_space: Configuration space definition
        
    Returns:
        bool: True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    for param_name, param_spec in config_space.items():
        if param_name not in config:
            # Skip optional parameters
            continue
            
        value = config[param_name]
        param_type = param_spec.get("type", "unknown")
        
        # Type validation
        if param_type == "int" and not isinstance(value, (int, np.integer)):
            raise ValidationError(f"Parameter {param_name} must be int, got {type(value)}")
        elif param_type == "float" and not isinstance(value, (int, float, np.number)):
            raise ValidationError(f"Parameter {param_name} must be float, got {type(value)}")
        elif param_type == "categorical" and value not in param_spec.get("choices", []):
            raise ValidationError(f"Parameter {param_name} must be one of {param_spec['choices']}, got {value}")
        
        # Range validation
        if param_type in ["int", "float"] and "bounds" in param_spec:
            bounds = param_spec["bounds"]
            if not (bounds[0] <= value <= bounds[1]):
                raise ValidationError(f"Parameter {param_name} must be in range {bounds}, got {value}")
    
    return True


def create_searcher(searcher_type: str, learner_names: List[str], task: str = "classification",
                   **kwargs) -> BaseSearcher:
    """
    Factory function to create searchers.
    
    Args:
        searcher_type: Type of searcher ("optuna", "flaml", "random")
        learner_names: List of learner names to search over
        task: Task type
        **kwargs: Additional arguments for searcher
        
    Returns:
        BaseSearcher: Configured searcher instance
    """
    # Import here to avoid circular imports
    if searcher_type.lower() == "optuna":
        from .optuna_search import OptunaSearcher
        return OptunaSearcher(learner_names, task, **kwargs)
    elif searcher_type.lower() == "flaml":
        from .flaml_search import FLAMLSearcher  
        return FLAMLSearcher(learner_names, task, **kwargs)
    elif searcher_type.lower() == "random":
        from .random_search import RandomSearcher
        return RandomSearcher(learner_names, task, **kwargs)
    else:
        available = ["optuna", "flaml", "random"]
        raise ValidationError(f"Unknown searcher type: {searcher_type}. Available: {available}")
