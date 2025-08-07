"""
Phase 5: AutoML Controller System for MLTeammate

Clean, modern AutoML orchestration that seamlessly integrates all frozen components:
- Phase 1: Utilities (logging, validation, timing)  
- Phase 2: Data processing (preprocessing, validation, resampling)
- Phase 3: Learner registry (8 learners, config spaces)
- Phase 4: Search system (multiple algorithms, validation)

This module provides:
- Complete AutoML orchestration
- Flexible callback system (MLflow, progress, artifacts)
- Cross-validation support
- Error handling and recovery
- Comprehensive result tracking
- Factory pattern for easy controller creation
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
import uuid
import time
from sklearn.model_selection import cross_val_score, cross_val_predict

# Import our frozen Phase 1 utilities
from ml_teammate.utils import (
    get_logger, ValidationError, Timer, time_context,
    MetricCalculator, ensure_numpy_array
)
# Import our frozen Phase 2 data processing
from ml_teammate.data import (
    DataPreprocessor, DataValidator, DataResampler, OverfitMonitor
)

# Import our frozen Phase 3 learner registry
from ml_teammate.learners.registry import (
    get_learner_registry, create_learner, get_learner_config_space
)

# Import our frozen Phase 4 search system
from ml_teammate.search import (
    create_searcher_by_name, get_available_searchers
)


class AutoMLController:
    """
    Central AutoML controller that orchestrates the complete machine learning pipeline.
    
    Integrates all frozen MLTeammate components into a cohesive AutoML system
    with flexible configuration, robust error handling, and comprehensive tracking.
    """
    
    def __init__(self, learner_names: List[str], task: str = "classification",
                 searcher_type: str = "random", n_trials: int = 10,
                 cv_folds: Optional[int] = None, random_state: int = 42,
                 callbacks: Optional[List] = None, log_level: str = "INFO"):
        """
        Initialize AutoML controller.
        
        Args:
            learner_names: List of learner names to use
            task: Task type ("classification" or "regression")
            searcher_type: Search algorithm ("random", "optuna", "flaml")
            n_trials: Number of optimization trials
            cv_folds: Cross-validation folds (None for train/test split)
            random_state: Random seed for reproducibility
            callbacks: List of callback objects
            log_level: Logging level
        """
        self.learner_names = learner_names
        self.task = task
        self.searcher_type = searcher_type
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.callbacks = callbacks or []
        
        # Initialize logger
        self.logger = get_logger(f"AutoMLController", log_level)
        
        # Validate inputs
        self._validate_configuration()
        
        # Initialize components
        self.data_preprocessor = DataPreprocessor(task=task)
        self.data_validator = DataValidator()
        self.metric_calculator = MetricCalculator()
        
        # Initialize searcher
        self.searcher = create_searcher_by_name(
            searcher_type, learner_names, task, random_state=random_state
        )
        
        # Results tracking
        self.results = {
            "trials": [],
            "best_score": None,
            "best_config": None,
            "best_model": None,
            "experiment_info": {},
            "search_history": []
        }
        
        # State tracking
        self.is_fitted = False
        self.experiment_start_time = None
        self.preprocessed_data = None
        
        self.logger.info(f"AutoMLController initialized: {len(learner_names)} learners, "
                        f"{n_trials} trials, {searcher_type} search")
    
    def _validate_configuration(self):
        """Validate controller configuration."""
        if not self.learner_names:
            raise ValidationError("learner_names cannot be empty")
        
        if self.task not in ["classification", "regression"]:
            raise ValidationError(f"Invalid task: {self.task}")
        
        if self.n_trials <= 0:
            raise ValidationError("n_trials must be positive")
        
        if self.cv_folds is not None and self.cv_folds < 2:
            raise ValidationError("cv_folds must be >= 2 if specified")
        
        # Validate searcher type
        available_searchers = get_available_searchers()
        if self.searcher_type not in available_searchers:
            raise ValidationError(f"Invalid searcher: {self.searcher_type}. "
                                f"Available: {available_searchers}")
        
        # Validate learners exist for task
        registry = get_learner_registry()
        available_learners = registry.list_learners(self.task)
        
        for learner_name in self.learner_names:
            if learner_name not in available_learners:
                available = list(available_learners.keys())
                raise ValidationError(f"Learner '{learner_name}' not available for {self.task}. "
                                    f"Available: {available}")
    
    def fit(self, X, y, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> 'AutoMLController':
        """
        Fit AutoML system to training data.
        
        Args:
            X: Training features
            y: Training targets
            X_val: Optional validation features
            y_val: Optional validation targets
            
        Returns:
            self: Fitted controller
        """
        with time_context("AutoMLController.fit") as experiment_timer:
            self.experiment_start_time = time.time()
            
            # Validate and preprocess data
            X, y = self._prepare_data(X, y)
            
            if X_val is not None and y_val is not None:
                X_val, y_val = self._prepare_validation_data(X_val, y_val)
            
            # Prepare experiment configuration
            experiment_config = self._create_experiment_config(X, y)
            
            # Notify callbacks of experiment start
            self._notify_callbacks("on_experiment_start", experiment_config)
            
            # Run optimization trials
            self._run_optimization_trials(X, y, X_val, y_val)
            
            # Finalize results
            self._finalize_experiment(experiment_timer.get_elapsed())
            
            # Notify callbacks of experiment end
            self._notify_callbacks("on_experiment_end", self.results)
            
            self.is_fitted = True
            
            # Safe formatting for best_score (handles None when all trials fail)
            best_score_str = f"{self.results['best_score']:.4f}" if self.results['best_score'] is not None else "N/A"
            self.logger.info(f"AutoML completed: best score = {best_score_str}, "
                           f"total time = {experiment_timer.get_elapsed():.2f}s")
        
        return self
    
    def _prepare_data(self, X, y) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare and validate training data."""
        with time_context("AutoMLController.prepare_data") as timer:
            # Convert to numpy arrays
            X = ensure_numpy_array(X)
            y = ensure_numpy_array(y)
            
            # Store original data info
            self.results["experiment_info"]["original_data_shape"] = X.shape
            self.results["experiment_info"]["target_distribution"] = self._get_target_distribution(y)
            
            # Preprocess data using the actual frozen interface
            X_processed = self.data_preprocessor.fit_transform(X, y)
            y_processed = self.data_preprocessor.transform_target(y)
            
            # Store preprocessed data info for Phase 5 use
            self.preprocessed_data = {
                "X": X_processed,
                "y": y_processed,
                "original_shape": X.shape,
                "processed_shape": X_processed.shape
            }
            
            self.logger.info(f"Data prepared in {timer.get_elapsed():.3f}s: "
                           f"{X.shape} -> {X_processed.shape}")
            
            return X_processed, y_processed
    
    def _prepare_validation_data(self, X_val, y_val) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare validation data using fitted preprocessor."""
        X_val = ensure_numpy_array(X_val)
        y_val = ensure_numpy_array(y_val)
        
        # Transform using fitted preprocessor
        X_val_processed = self.data_preprocessor.transform(X_val)
        
        return X_val_processed, y_val
    
    def _create_experiment_config(self, X, y) -> Dict[str, Any]:
        """Create experiment configuration for callbacks."""
        return {
            "controller_id": str(uuid.uuid4()),
            "task": self.task,
            "learner_names": self.learner_names,
            "searcher_type": self.searcher_type,
            "n_trials": self.n_trials,
            "cv_folds": self.cv_folds,
            "random_state": self.random_state,
            "data_shape": X.shape,
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "target_type": str(y.dtype),
            "start_time": self.experiment_start_time
        }
    
    def _run_optimization_trials(self, X, y, X_val=None, y_val=None):
        """Run hyperparameter optimization trials."""
        self.logger.info(f"Starting {self.n_trials} optimization trials...")
        
        for trial_idx in range(self.n_trials):
            trial_id = f"trial_{trial_idx:03d}_{str(uuid.uuid4())[:8]}"
            
            try:
                with time_context(f"Trial_{trial_idx}") as trial_timer:
                    # Get configuration suggestion
                    config = self.searcher.suggest(trial_id)
                    
                    # Notify callbacks of trial start
                    self._notify_callbacks("on_trial_start", trial_id, config)
                    
                    # Evaluate configuration
                    score = self._evaluate_configuration(config, X, y, X_val, y_val)
                    
                    # Report result to searcher
                    self.searcher.report(trial_id, score, config)
                    
                    # Track trial result
                    trial_result = {
                        "trial_id": trial_id,
                        "trial_idx": trial_idx,
                        "config": config.copy(),
                        "score": score,
                        "duration": trial_timer.get_elapsed(),
                        "timestamp": time.time()
                    }
                    
                    self.results["trials"].append(trial_result)
                    self.results["search_history"].append(score)
                    
                    # Update best result
                    is_best = self._update_best_result(config, score)
                    trial_result["is_best"] = is_best
                    
                    # Notify callbacks of trial end
                    self._notify_callbacks("on_trial_end", trial_id, config, score, is_best)
                    
                    self.logger.info(f"Trial {trial_idx+1}/{self.n_trials}: "
                                   f"{config['learner_name']} -> {score:.4f} "
                                   f"({'BEST' if is_best else 'ok'})")
                    
            except Exception as e:
                self.logger.warning(f"Trial {trial_idx+1} failed: {str(e)}")
                # Continue with next trial
                continue
    
    def _evaluate_configuration(self, config: Dict[str, Any], X, y, X_val=None, y_val=None) -> float:
        """Evaluate a single configuration."""
        # Create learner from configuration
        learner = create_learner(config["learner_name"], 
                               {k: v for k, v in config.items() if k != "learner_name"})
        
        if self.cv_folds is not None:
            # Manual cross-validation to avoid sklearn clone issues with SklearnWrapper
            scores = self._manual_cross_validation(config, X, y)
            score = np.mean(scores)
        
        elif X_val is not None and y_val is not None:
            # Validation set evaluation
            learner.fit(X, y)
            y_pred = learner.predict(X_val)
            score = self.metric_calculator.calculate_score(y_val, y_pred, self.task)
        
        else:
            # Training set evaluation (not recommended for final results)
            learner.fit(X, y)
            y_pred = learner.predict(X)
            score = self.metric_calculator.calculate_score(y, y_pred, self.task)
        
        return score
    
    def _manual_cross_validation(self, config: Dict[str, Any], X, y) -> List[float]:
        """
        Manual cross-validation implementation to avoid sklearn clone issues.
        
        Creates fresh learner instances for each fold instead of cloning.
        """
        from sklearn.model_selection import KFold, StratifiedKFold
        
        # Choose CV strategy
        if self.task == "classification":
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        else:
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            # Split data for this fold
            X_train_fold = X[train_idx]
            X_val_fold = X[val_idx]
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]
            
            # Create fresh learner instance (no cloning needed)
            learner = create_learner(config["learner_name"], 
                                   {k: v for k, v in config.items() if k != "learner_name"})
            
            # Train and evaluate on this fold
            learner.fit(X_train_fold, y_train_fold)
            y_pred_fold = learner.predict(X_val_fold)
            fold_score = self.metric_calculator.calculate_score(y_val_fold, y_pred_fold, self.task)
            
            scores.append(fold_score)
        
        return scores
    
    def _update_best_result(self, config: Dict[str, Any], score: float) -> bool:
        """Update best result if current is better."""
        is_best = (self.results["best_score"] is None or 
                  score > self.results["best_score"])
        
        if is_best:
            self.results["best_score"] = score
            self.results["best_config"] = config.copy()
            
            # Create and store best model
            self.results["best_model"] = create_learner(
                config["learner_name"],
                {k: v for k, v in config.items() if k != "learner_name"}
            )
        
        return is_best
    
    def _finalize_experiment(self, total_time: float):
        """Finalize experiment results."""
        # Fit best model on all data
        if self.results["best_model"] is not None and self.preprocessed_data is not None:
            X_processed = self.preprocessed_data["X"]
            y_processed = self.preprocessed_data["y"]
            self.results["best_model"].fit(X_processed, y_processed)
        
        # Update experiment info
        self.results["experiment_info"].update({
            "total_time": total_time,
            "n_completed_trials": len(self.results["trials"]),
            "success_rate": len(self.results["trials"]) / self.n_trials,
            "best_learner": self.results["best_config"]["learner_name"] if self.results["best_config"] else None,
            "score_improvement": self._calculate_score_improvement(),
            "searcher_summary": self.searcher.get_search_summary()
        })
    
    def predict(self, X) -> np.ndarray:
        """Make predictions with the best model."""
        if not self.is_fitted:
            raise ValidationError("Controller not fitted. Call fit() first.")
        
        if self.results["best_model"] is None:
            raise ValidationError("No successful trials found. Cannot make predictions.")
        
        # Preprocess input data
        X = ensure_numpy_array(X)
        X_processed = self.data_preprocessor.transform(X)
        
        # Make predictions
        return self.results["best_model"].predict(X_processed)
    
    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities (classification only)."""
        if self.task != "classification":
            raise ValidationError("predict_proba only available for classification")
        
        if not self.is_fitted:
            raise ValidationError("Controller not fitted. Call fit() first.")
        
        if self.results["best_model"] is None:
            raise ValidationError("No successful trials found. Cannot make predictions.")
        
        # Preprocess input data
        X = ensure_numpy_array(X)
        X_processed = self.data_preprocessor.transform(X)
        
        # Make probability predictions
        return self.results["best_model"].predict_proba(X_processed)
    
    def score(self, X, y) -> float:
        """Evaluate the best model on test data."""
        y_pred = self.predict(X)
        y = ensure_numpy_array(y)
        return self.metric_calculator.calculate_score(y, y_pred, self.task)
    
    def get_results(self) -> Dict[str, Any]:
        """Get comprehensive experiment results."""
        return self.results.copy()
    
    def get_best_model(self):
        """Get the best trained model."""
        if not self.is_fitted:
            raise ValidationError("Controller not fitted. Call fit() first.")
        
        return self.results["best_model"]
    
    def get_search_history(self) -> List[float]:
        """Get score history for plotting optimization progress."""
        return self.results["search_history"].copy()
    
    # Helper methods
    
    def _get_target_distribution(self, y) -> Dict[str, Any]:
        """Get target variable distribution info."""
        if self.task == "classification":
            unique_values, counts = np.unique(y, return_counts=True)
            return {
                "n_classes": len(unique_values),
                "class_distribution": dict(zip(unique_values.tolist(), counts.tolist())),
                "is_balanced": np.std(counts) / np.mean(counts) < 0.1
            }
        else:
            return {
                "mean": float(np.mean(y)),
                "std": float(np.std(y)),
                "min": float(np.min(y)),
                "max": float(np.max(y))
            }
    
    def _get_scoring_metric(self) -> str:
        """Get sklearn-compatible scoring metric name."""
        if self.task == "classification":
            return "accuracy"
        else:
            return "neg_mean_squared_error"
    
    def _calculate_score_improvement(self) -> float:
        """Calculate improvement from first to best score."""
        if not self.results["search_history"]:
            return 0.0
        
        first_score = self.results["search_history"][0]
        best_score = self.results["best_score"] or first_score
        
        return best_score - first_score
    
    def _notify_callbacks(self, method_name: str, *args, **kwargs):
        """Notify all callbacks of an event."""
        for callback in self.callbacks:
            if hasattr(callback, method_name):
                try:
                    getattr(callback, method_name)(*args, **kwargs)
                except Exception as e:
                    self.logger.warning(f"Callback {callback.__class__.__name__} failed: {e}")


def create_automl_controller(learner_names: List[str], task: str = "classification",
                           **kwargs) -> AutoMLController:
    """
    Factory function to create AutoML controller.
    
    Args:
        learner_names: List of learner names to use
        task: Task type ("classification" or "regression")
        **kwargs: Additional arguments for AutoMLController
        
    Returns:
        AutoMLController: Configured controller instance
    """
    return AutoMLController(learner_names, task, **kwargs)
