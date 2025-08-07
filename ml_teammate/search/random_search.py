"""
Random Search Implementation for MLTeammate

Provides baseline random hyperparameter search for comparison with more
sophisticated optimization algorithms.
"""

import numpy as np
import random
from typing import Dict, Any, Optional, List
import uuid
import time

from .base import BaseSearcher, SearchResult, validate_config_against_space
from ml_teammate.utils import get_logger, ValidationError, time_context


class RandomSearcher(BaseSearcher):
    """
    Random hyperparameter searcher for baseline comparisons.
    
    Randomly samples from the configuration space of each learner.
    Useful for establishing baseline performance and debugging.
    """
    
    def __init__(self, learner_names: List[str], task: str = "classification",
                 random_state: int = 42, log_level: str = "INFO"):
        """
        Initialize random searcher.
        
        Args:
            learner_names: List of learner names to search over
            task: Task type ("classification" or "regression")
            random_state: Random seed for reproducibility
            log_level: Logging level
        """
        # Initialize base searcher
        super().__init__(learner_names, task, random_state, log_level)
        
        # Set random seeds
        random.seed(random_state)
        np.random.seed(random_state)
        
        self.logger.info(f"RandomSearcher initialized with {len(learner_names)} learners")
    
    def suggest(self, trial_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Suggest next configuration using random sampling.
        
        Args:
            trial_id: Optional trial identifier
            
        Returns:
            dict: Configuration with 'learner_name' and hyperparameters
        """
        if trial_id is None:
            trial_id = str(uuid.uuid4())
        
        with time_context(f"RandomSearcher.suggest") as timer:
            # Randomly choose learner
            learner_name = random.choice(self.learner_names)
            
            # Get config space for the learner
            config_space = self.config_spaces[learner_name]
            
            # Randomly sample hyperparameters
            config = {"learner_name": learner_name}
            
            for param_name, param_spec in config_space.items():
                param_type = param_spec.get("type", "unknown")
                
                if param_type == "int":
                    bounds = param_spec["bounds"]
                    config[param_name] = random.randint(bounds[0], bounds[1])
                
                elif param_type == "float":
                    bounds = param_spec["bounds"]
                    log_scale = param_spec.get("log_scale", False)
                    
                    if log_scale:
                        # Sample in log space
                        log_low = np.log(bounds[0])
                        log_high = np.log(bounds[1])
                        log_value = random.uniform(log_low, log_high)
                        config[param_name] = np.exp(log_value)
                    else:
                        config[param_name] = random.uniform(bounds[0], bounds[1])
                
                elif param_type == "categorical":
                    choices = param_spec["choices"]
                    config[param_name] = random.choice(choices)
                
                elif param_type == "fixed":
                    config[param_name] = param_spec["value"]
                
                else:
                    self.logger.warning(f"Unknown parameter type: {param_type} for {param_name}")
            
            # Validate configuration
            validate_config_against_space(config, config_space)
            
            self.trials[trial_id] = {
                "config": config,
                "trial_id": trial_id,
                "status": "suggested",
                "suggest_time": timer.get_elapsed()
            }
            
            self.n_trials += 1
            
            self.logger.debug(f"Suggested random config for trial {trial_id}: {learner_name} "
                            f"with {len(config)-1} hyperparameters")
            
            return config
    
    def report(self, trial_id: str, score: float, config: Dict[str, Any]) -> None:
        """
        Report trial result.
        
        Args:
            trial_id: Trial identifier
            score: Performance score (higher is better)
            config: Configuration that was evaluated
        """
        with time_context(f"RandomSearcher.report") as timer:
            # Update our tracking
            if trial_id in self.trials:
                self.trials[trial_id].update({
                    "score": score,
                    "status": "completed",
                    "report_time": timer.get_elapsed()
                })
            
            # Update best score
            if self.best_score is None or score > self.best_score:
                self.best_score = score
                self.best_config = config.copy()
            
            self.logger.debug(f"Reported score {score:.4f} for trial {trial_id}")
    
    def create_search_result(self, search_time: float) -> SearchResult:
        """Create SearchResult from current state."""
        
        # Get all completed trials
        all_trials = []
        for trial_id, trial_info in self.trials.items():
            if trial_info.get("status") == "completed":
                all_trials.append(trial_info.copy())
        
        # Get searcher info
        searcher_info = {
            "searcher_type": "Random",
            "n_learners": len(self.learner_names),
            "random_state": self.random_state,
            "search_strategy": "uniform_random_sampling"
        }
        
        return SearchResult(
            best_config=self.best_config,
            best_score=self.best_score or 0.0,
            all_trials=all_trials,
            search_time=search_time,
            searcher_info=searcher_info
        )


def create_random_searcher(learner_names: List[str], task: str = "classification",
                          **kwargs) -> RandomSearcher:
    """
    Convenience function to create RandomSearcher.
    
    Args:
        learner_names: List of learner names
        task: Task type
        **kwargs: Additional arguments
        
    Returns:
        RandomSearcher: Configured searcher
    """
    return RandomSearcher(learner_names, task, **kwargs)
