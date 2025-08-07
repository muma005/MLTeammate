"""
Optuna Search Integration for MLTeammate

Provides comprehensive Optuna-based hyperparameter optimization with all major samplers,
pruning strategies, and multi-objective optimization support.
"""

import optuna
from optuna.samplers import (
    TPESampler, RandomSampler, CmaEsSampler, NSGAIISampler,
    GridSampler, PartiallyFixedSampler
)
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner, HyperbandPruner
from typing import Dict, Any, Optional, List, Union
import uuid
import time

from .base import BaseSearcher, SearchResult, validate_config_against_space
from ml_teammate.utils import get_logger, ValidationError, time_context


class OptunaSearcher(BaseSearcher):
    """
    Optuna-based hyperparameter searcher with comprehensive sampler support.
    
    Supports all major Optuna samplers and pruning strategies for efficient
    hyperparameter optimization.
    """
    
    def __init__(self, learner_names: List[str], task: str = "classification",
                 sampler: str = "TPE", pruner: Optional[str] = None,
                 direction: str = "maximize", random_state: int = 42,
                 sampler_kwargs: Optional[Dict[str, Any]] = None,
                 pruner_kwargs: Optional[Dict[str, Any]] = None,
                 log_level: str = "INFO"):
        """
        Initialize Optuna searcher.
        
        Args:
            learner_names: List of learner names to search over
            task: Task type ("classification" or "regression")
            sampler: Sampler type ("TPE", "Random", "CmaEs", "NSGAII", "Grid")
            pruner: Pruner type ("Median", "SuccessiveHalving", "Hyperband", None)
            direction: Optimization direction ("maximize" or "minimize")
            random_state: Random seed
            sampler_kwargs: Additional sampler arguments
            pruner_kwargs: Additional pruner arguments
            log_level: Logging level
        """
        # Initialize base searcher first
        super().__init__(learner_names, task, random_state, log_level)
        
        self.sampler_type = sampler
        self.pruner_type = pruner
        self.direction = direction
        self.sampler_kwargs = sampler_kwargs or {}
        self.pruner_kwargs = pruner_kwargs or {}
        
        # Create Optuna study
        self.study = self._create_study()
        
        # Trial tracking
        self.optuna_trials = {}  # trial_id -> optuna.Trial
        
        self.logger.info(f"OptunaSearcher initialized with {sampler} sampler, "
                        f"{pruner or 'no'} pruner, direction: {direction}")
    
    def _create_study(self) -> optuna.Study:
        """Create Optuna study with configured sampler and pruner."""
        
        # Create sampler
        sampler = self._create_sampler()
        
        # Create pruner
        pruner = self._create_pruner()
        
        # Create study
        study = optuna.create_study(
            direction=self.direction,
            sampler=sampler,
            pruner=pruner
        )
        
        return study
    
    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """Create Optuna sampler based on configuration."""
        
        # Add random state to sampler kwargs if not present
        if "seed" not in self.sampler_kwargs:
            self.sampler_kwargs["seed"] = self.random_state
        
        if self.sampler_type.upper() == "TPE":
            return TPESampler(**self.sampler_kwargs)
        elif self.sampler_type.upper() == "RANDOM":
            return RandomSampler(**self.sampler_kwargs)
        elif self.sampler_type.upper() == "CMAES":
            return CmaEsSampler(**self.sampler_kwargs)
        elif self.sampler_type.upper() == "NSGAII":
            return NSGAIISampler(**self.sampler_kwargs)
        elif self.sampler_type.upper() == "GRID":
            # Grid sampler requires search space definition
            return GridSampler(**self.sampler_kwargs)
        else:
            self.logger.warning(f"Unknown sampler type: {self.sampler_type}, using TPE")
            return TPESampler(**self.sampler_kwargs)
    
    def _create_pruner(self) -> Optional[optuna.pruners.BasePruner]:
        """Create Optuna pruner based on configuration."""
        
        if not self.pruner_type:
            return None
        
        if self.pruner_type.upper() == "MEDIAN":
            return MedianPruner(**self.pruner_kwargs)
        elif self.pruner_type.upper() == "SUCCESSIVEHALVING":
            return SuccessiveHalvingPruner(**self.pruner_kwargs)
        elif self.pruner_type.upper() == "HYPERBAND":
            return HyperbandPruner(**self.pruner_kwargs)
        else:
            self.logger.warning(f"Unknown pruner type: {self.pruner_type}, using no pruner")
            return None
    
    def suggest(self, trial_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Suggest next configuration using Optuna.
        
        Args:
            trial_id: Optional trial identifier
            
        Returns:
            dict: Configuration with 'learner_name' and hyperparameters
        """
        if trial_id is None:
            trial_id = str(uuid.uuid4())
        
        # Create Optuna trial
        trial = self.study.ask()
        self.optuna_trials[trial_id] = trial
        
        with time_context(f"OptunaSearcher.suggest") as timer:
            # Suggest learner
            learner_name = trial.suggest_categorical("learner_name", self.learner_names)
            
            # Get config space for the learner
            config_space = self.config_spaces[learner_name]
            
            # Suggest hyperparameters
            config = {"learner_name": learner_name}
            
            for param_name, param_spec in config_space.items():
                param_type = param_spec.get("type", "unknown")
                
                if param_type == "int":
                    bounds = param_spec["bounds"]
                    step = param_spec.get("step", 1)
                    config[param_name] = trial.suggest_int(param_name, bounds[0], bounds[1], step=step)
                
                elif param_type == "float":
                    bounds = param_spec["bounds"]
                    log_scale = param_spec.get("log_scale", False)
                    step = param_spec.get("step", None)
                    config[param_name] = trial.suggest_float(
                        param_name, bounds[0], bounds[1], log=log_scale, step=step
                    )
                
                elif param_type == "categorical":
                    choices = param_spec["choices"]
                    config[param_name] = trial.suggest_categorical(param_name, choices)
                
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
            
            self.logger.debug(f"Suggested config for trial {trial_id}: {learner_name} "
                            f"with {len(config)-1} hyperparameters")
            
            return config
    
    def report(self, trial_id: str, score: float, config: Dict[str, Any]) -> None:
        """
        Report trial result to Optuna.
        
        Args:
            trial_id: Trial identifier
            score: Performance score (higher is better for maximize)
            config: Configuration that was evaluated
        """
        if trial_id not in self.optuna_trials:
            raise ValidationError(f"Trial {trial_id} not found in optuna_trials")
        
        trial = self.optuna_trials[trial_id]
        
        with time_context(f"OptunaSearcher.report") as timer:
            # Report to Optuna
            self.study.tell(trial, score)
            
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
    
    def prune_trial(self, trial_id: str, intermediate_score: float, step: int) -> bool:
        """
        Check if trial should be pruned based on intermediate score.
        
        Args:
            trial_id: Trial identifier
            intermediate_score: Intermediate performance score
            step: Current step/epoch
            
        Returns:
            bool: True if trial should be pruned
        """
        if trial_id not in self.optuna_trials:
            return False
        
        trial = self.optuna_trials[trial_id]
        
        # Report intermediate value
        trial.report(intermediate_score, step)
        
        # Check if should prune
        should_prune = trial.should_prune()
        
        if should_prune:
            self.logger.debug(f"Pruning trial {trial_id} at step {step} with score {intermediate_score:.4f}")
            
            # Update trial status
            if trial_id in self.trials:
                self.trials[trial_id]["status"] = "pruned"
        
        return should_prune
    
    def get_optimization_history(self) -> List[float]:
        """Get optimization history (best score over time)."""
        if not self.study.trials:
            return []
        
        best_scores = []
        current_best = None
        
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                score = trial.value
                if current_best is None or score > current_best:
                    current_best = score
                best_scores.append(current_best)
            else:
                if current_best is not None:
                    best_scores.append(current_best)
        
        return best_scores
    
    def get_parameter_importance(self) -> Dict[str, float]:
        """Get parameter importance using Optuna's analysis."""
        try:
            importance = optuna.importance.get_param_importances(self.study)
            return importance
        except Exception as e:
            self.logger.warning(f"Failed to compute parameter importance: {e}")
            return {}
    
    def create_search_result(self, search_time: float) -> SearchResult:
        """Create SearchResult from current state."""
        
        # Get all completed trials
        all_trials = []
        for trial_id, trial_info in self.trials.items():
            if trial_info.get("status") == "completed":
                all_trials.append(trial_info.copy())
        
        # Get searcher info
        searcher_info = {
            "searcher_type": "Optuna",
            "sampler": self.sampler_type,
            "pruner": self.pruner_type,
            "direction": self.direction,
            "n_learners": len(self.learner_names),
            "parameter_importance": self.get_parameter_importance(),
            "optimization_history": self.get_optimization_history()
        }
        
        return SearchResult(
            best_config=self.best_config,
            best_score=self.best_score or 0.0,
            all_trials=all_trials,
            search_time=search_time,
            searcher_info=searcher_info
        )


def create_optuna_searcher(learner_names: List[str], task: str = "classification",
                          sampler: str = "TPE", **kwargs) -> OptunaSearcher:
    """
    Convenience function to create OptunaSearcher.
    
    Args:
        learner_names: List of learner names
        task: Task type
        sampler: Sampler type
        **kwargs: Additional arguments
        
    Returns:
        OptunaSearcher: Configured searcher
    """
    return OptunaSearcher(learner_names, task, sampler=sampler, **kwargs)
