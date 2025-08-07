"""
FLAML Search Integration for MLTeammate

Provides FLAML-based hyperparameter optimization with time-budget constraints
and efficient search strategies.
"""

from typing import Dict, Any, Optional, List, Union
import uuid
import time

from .base import BaseSearcher, SearchResult, validate_config_against_space
from ml_teammate.utils import get_logger, ValidationError, time_context

try:
    import flaml
    FLAML_AVAILABLE = True
except ImportError:
    FLAML_AVAILABLE = False


class FLAMLSearcher(BaseSearcher):
    """
    FLAML-based hyperparameter searcher with time-budget optimization.
    
    Uses FLAML's efficient search strategies for cost-effective hyperparameter
    optimization with resource constraints.
    """
    
    def __init__(self, learner_names: List[str], task: str = "classification",
                 time_budget: float = 60.0, max_iter: Optional[int] = None,
                 random_state: int = 42, log_level: str = "INFO"):
        """
        Initialize FLAML searcher.
        
        Args:
            learner_names: List of learner names to search over
            task: Task type ("classification" or "regression")
            time_budget: Time budget in seconds
            max_iter: Maximum number of iterations
            random_state: Random seed
            log_level: Logging level
        """
        if not FLAML_AVAILABLE:
            raise ValidationError("FLAML is not available. Install with: pip install flaml")
        
        # Initialize base searcher
        super().__init__(learner_names, task, random_state, log_level)
        
        self.time_budget = time_budget
        self.max_iter = max_iter
        
        # FLAML configuration
        self.flaml_config_space = self._build_flaml_config_space()
        
        # Initialize FLAML searcher using AutoML interface
        self.flaml_searcher = flaml.AutoML(
            time_budget=time_budget,
            max_iter=max_iter,
            task=task,
            seed=random_state
        )
        
        self.search_started = False
        self.start_time = None
        
        self.logger.info(f"FLAMLSearcher initialized with {time_budget}s budget")
    
    def _build_flaml_config_space(self) -> Dict[str, Any]:
        """Build FLAML-compatible configuration space."""
        flaml_space = {}
        
        # Add learner choice
        flaml_space["learner_name"] = flaml.tune.choice(self.learner_names)
        
        # Add hyperparameters for each learner
        for learner_name, config_space in self.config_spaces.items():
            for param_name, param_spec in config_space.items():
                param_type = param_spec.get("type", "unknown")
                flaml_param_name = f"{learner_name}__{param_name}"
                
                if param_type == "int":
                    bounds = param_spec["bounds"]
                    flaml_space[flaml_param_name] = flaml.tune.randint(bounds[0], bounds[1] + 1)
                
                elif param_type == "float":
                    bounds = param_spec["bounds"]
                    log_scale = param_spec.get("log_scale", False)
                    if log_scale:
                        flaml_space[flaml_param_name] = flaml.tune.loguniform(bounds[0], bounds[1])
                    else:
                        flaml_space[flaml_param_name] = flaml.tune.uniform(bounds[0], bounds[1])
                
                elif param_type == "categorical":
                    choices = param_spec["choices"]
                    flaml_space[flaml_param_name] = flaml.tune.choice(choices)
                
                elif param_type == "fixed":
                    flaml_space[flaml_param_name] = param_spec["value"]
        
        return flaml_space
    
    def _extract_config_from_flaml(self, flaml_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract MLTeammate config from FLAML configuration."""
        learner_name = flaml_config["learner_name"]
        config = {"learner_name": learner_name}
        
        # Extract hyperparameters for the selected learner
        prefix = f"{learner_name}__"
        for key, value in flaml_config.items():
            if key.startswith(prefix):
                param_name = key[len(prefix):]
                config[param_name] = value
        
        return config
    
    def suggest(self, trial_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Suggest next configuration using FLAML.
        
        Args:
            trial_id: Optional trial identifier
            
        Returns:
            dict: Configuration with 'learner_name' and hyperparameters
        """
        if trial_id is None:
            trial_id = str(uuid.uuid4())
        
        if not self.search_started:
            self.start_time = time.time()
            self.search_started = True
        
        with time_context(f"FLAMLSearcher.suggest") as timer:
            # Get suggestion from FLAML
            flaml_config = self.flaml_searcher.suggest(trial_id)
            
            # Convert to MLTeammate format
            config = self._extract_config_from_flaml(flaml_config)
            
            # Validate configuration
            learner_name = config["learner_name"]
            config_space = self.config_spaces[learner_name]
            validate_config_against_space(config, config_space)
            
            self.trials[trial_id] = {
                "config": config,
                "flaml_config": flaml_config,
                "trial_id": trial_id,
                "status": "suggested",
                "suggest_time": timer.get_elapsed()
            }
            
            self.n_trials += 1
            
            self.logger.debug(f"Suggested config for trial {trial_id}: {learner_name}")
            
            return config
    
    def report(self, trial_id: str, score: float, config: Dict[str, Any]) -> None:
        """
        Report trial result to FLAML.
        
        Args:
            trial_id: Trial identifier
            score: Performance score (higher is better)
            config: Configuration that was evaluated
        """
        if trial_id not in self.trials:
            raise ValidationError(f"Trial {trial_id} not found in trials")
        
        with time_context(f"FLAMLSearcher.report") as timer:
            # Report to FLAML
            self.flaml_searcher.update(trial_id, {"score": score})
            
            # Update our tracking
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
    
    def should_stop(self) -> bool:
        """Check if search should stop based on time budget or max iterations."""
        if self.start_time is None:
            return False
        
        # Check time budget
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= self.time_budget:
            return True
        
        # Check max iterations
        if self.max_iter is not None and self.n_trials >= self.max_iter:
            return True
        
        return False
    
    def create_search_result(self, search_time: float) -> SearchResult:
        """Create SearchResult from current state."""
        
        # Get all completed trials
        all_trials = []
        for trial_id, trial_info in self.trials.items():
            if trial_info.get("status") == "completed":
                all_trials.append(trial_info.copy())
        
        # Get searcher info
        searcher_info = {
            "searcher_type": "FLAML",
            "time_budget": self.time_budget,
            "max_iter": self.max_iter,
            "n_learners": len(self.learner_names),
            "actual_time": search_time,
            "time_efficiency": min(1.0, search_time / self.time_budget) if self.time_budget > 0 else 1.0
        }
        
        return SearchResult(
            best_config=self.best_config,
            best_score=self.best_score or 0.0,
            all_trials=all_trials,
            search_time=search_time,
            searcher_info=searcher_info
        )


def create_flaml_searcher(learner_names: List[str], task: str = "classification",
                         time_budget: float = 60.0, **kwargs) -> FLAMLSearcher:
    """
    Convenience function to create FLAMLSearcher.
    
    Args:
        learner_names: List of learner names
        task: Task type
        time_budget: Time budget in seconds
        **kwargs: Additional arguments
        
    Returns:
        FLAMLSearcher: Configured searcher
    """
    return FLAMLSearcher(learner_names, task, time_budget=time_budget, **kwargs)
