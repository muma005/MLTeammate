"""
Simplified API for MLTeammate

This module provides an extremely user-friendly interface that eliminates the need
for users to write custom classes, functions, or configuration spaces.

Users can simply specify learner names as strings and the framework handles everything else.
"""

from typing import List, Dict, Any, Optional, Union
from ml_teammate.learners.registry import (
    create_learners_dict,
    create_config_space,
    get_all_learners,
    get_classification_learners,
    get_regression_learners
)
from ml_teammate.automl.controller import AutoMLController
from ml_teammate.search.optuna_search import OptunaSearcher
from ml_teammate.automl.callbacks import LoggerCallback, ProgressCallback, ArtifactCallback


class SimpleAutoML:
    """
    Simplified AutoML interface that requires no custom code from users.
    
    Users can simply specify learner names as strings and the framework
    automatically handles all the complexity behind the scenes.
    """
    
    def __init__(self, 
                 learners: Union[List[str], str] = None,
                 task: str = "classification",
                 n_trials: int = 10,
                 cv: Optional[int] = None,
                 use_mlflow: bool = False,
                 experiment_name: str = "mlteammate_experiment",
                 log_level: str = "INFO",
                 save_artifacts: bool = True,
                 output_dir: str = "./mlteammate_artifacts"):
        """
        Initialize SimpleAutoML with minimal configuration.
        
        Args:
            learners: List of learner names or single learner name as string
                     Examples: ["random_forest", "logistic_regression", "xgboost"]
                              or "random_forest" for single learner
            task: "classification" or "regression"
            n_trials: Number of hyperparameter optimization trials
            cv: Number of cross-validation folds (None for no CV)
            use_mlflow: Whether to use MLflow for experiment tracking
            experiment_name: Name for MLflow experiment
            log_level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
            save_artifacts: Whether to save models and plots
            output_dir: Directory to save artifacts
        """
        self.task = task
        self.n_trials = n_trials
        self.cv = cv
        self.use_mlflow = use_mlflow
        self.experiment_name = experiment_name
        self.log_level = log_level
        self.save_artifacts = save_artifacts
        self.output_dir = output_dir
        
        # Handle learners parameter
        if learners is None:
            # Use sensible defaults based on task
            if task == "classification":
                learners = ["random_forest", "logistic_regression", "xgboost"]
            else:
                learners = ["random_forest_regressor", "linear_regression", "ridge"]
        elif isinstance(learners, str):
            learners = [learners]
        
        # Validate learners
        self._validate_learners(learners, task)
        self.learner_names = learners
        
        # Create learners and config space automatically
        self.learners = create_learners_dict(learners)
        self.config_space = create_config_space(learners)
        
        # Create callbacks
        self.callbacks = self._create_callbacks()
        
        # Initialize controller
        self.controller = None
        self._create_controller()
    
    def _validate_learners(self, learners: List[str], task: str):
        """Validate that the specified learners are appropriate for the task."""
        available_learners = get_all_learners()
        
        for learner in learners:
            if learner not in available_learners:
                available = ", ".join(sorted(available_learners))
                raise ValueError(f"Unknown learner '{learner}'. Available learners: {available}")
        
        # Check task compatibility
        if task == "classification":
            classification_learners = get_classification_learners()
            incompatible = [l for l in learners if l not in classification_learners]
            if incompatible:
                print(f"⚠️ Warning: The following learners may not be optimal for classification: {incompatible}")
        elif task == "regression":
            regression_learners = get_regression_learners()
            incompatible = [l for l in learners if l not in regression_learners]
            if incompatible:
                print(f"⚠️ Warning: The following learners may not be optimal for regression: {incompatible}")
    
    def _create_callbacks(self):
        """Create appropriate callbacks based on configuration."""
        callbacks = [
            LoggerCallback(
                use_mlflow=self.use_mlflow,
                log_level=self.log_level,
                experiment_name=self.experiment_name
            ),
            ProgressCallback(total_trials=self.n_trials, patience=3)
        ]
        
        if self.save_artifacts:
            callbacks.append(ArtifactCallback(
                save_best_model=True,
                save_configs=True,
                output_dir=self.output_dir
            ))
        
        return callbacks
    
    def _create_controller(self):
        """Create the AutoML controller."""
        self.controller = AutoMLController(
            learners=self.learners,
            searcher=OptunaSearcher(self.config_space),
            config_space=self.config_space,
            task=self.task,
            n_trials=self.n_trials,
            cv=self.cv,
            callbacks=self.callbacks
        )
    
    def fit(self, X, y):
        """
        Fit the AutoML model to the data.
        
        Args:
            X: Training features
            y: Training targets
        
        Returns:
            self: For method chaining
        """
        print(f"🚀 Starting AutoML experiment with {len(self.learner_names)} learners:")
        for i, learner in enumerate(self.learner_names, 1):
            print(f"   {i}. {learner}")
        print(f"📊 Task: {self.task}")
        print(f"🔬 Trials: {self.n_trials}")
        if self.cv:
            print(f"🔄 Cross-validation: {self.cv} folds")
        print()
        
        self.controller.fit(X, y)
        return self
    
    def predict(self, X):
        """
        Make predictions using the best model.
        
        Args:
            X: Features to predict on
        
        Returns:
            Predictions
        """
        if self.controller is None or self.controller.best_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.controller.predict(X)
    
    def score(self, X, y):
        """
        Score the best model on the given data.
        
        Args:
            X: Features
            y: True targets
        
        Returns:
            Score (accuracy for classification, R² for regression)
        """
        if self.controller is None or self.controller.best_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.controller.score(X, y)
    
    @property
    def best_score(self):
        """Get the best cross-validation score."""
        if self.controller is None:
            return None
        return self.controller.best_score
    
    @property
    def best_model(self):
        """Get the best trained model."""
        if self.controller is None:
            return None
        return self.controller.best_model
    
    @property
    def best_config(self):
        """Get the best hyperparameter configuration."""
        if self.controller is None:
            return None
        return self.controller.searcher.get_best()
    
    def get_results_summary(self):
        """
        Get a summary of the experiment results.
        
        Returns:
            Dictionary with experiment summary
        """
        if self.controller is None:
            return None
        
        return {
            "task": self.task,
            "learners_used": self.learner_names,
            "n_trials": self.n_trials,
            "cv_folds": self.cv,
            "best_score": self.best_score,
            "best_learner": self.best_config.get("learner_name", "unknown") if self.best_config else None,
            "best_config": self.best_config,
            "mlflow_enabled": self.use_mlflow,
            "experiment_name": self.experiment_name
        }


# Convenience functions for even simpler usage
def quick_classification(X, y, 
                        learners: Union[List[str], str] = None,
                        n_trials: int = 10,
                        cv: int = 3,
                        **kwargs):
    """
    Quick classification experiment with minimal setup.
    
    Args:
        X: Training features
        y: Training targets
        learners: Learner names (defaults to ["random_forest", "logistic_regression", "xgboost"])
        n_trials: Number of trials
        cv: Cross-validation folds
        **kwargs: Additional arguments for SimpleAutoML
    
    Returns:
        SimpleAutoML instance with fitted model
    """
    if learners is None:
        learners = ["random_forest", "logistic_regression", "xgboost"]
    
    automl = SimpleAutoML(
        learners=learners,
        task="classification",
        n_trials=n_trials,
        cv=cv,
        **kwargs
    )
    
    automl.fit(X, y)
    return automl


def quick_regression(X, y,
                    learners: Union[List[str], str] = None,
                    n_trials: int = 10,
                    cv: int = 3,
                    **kwargs):
    """
    Quick regression experiment with minimal setup.
    
    Args:
        X: Training features
        y: Training targets
        learners: Learner names (defaults to ["random_forest_regressor", "linear_regression", "ridge"])
        n_trials: Number of trials
        cv: Cross-validation folds
        **kwargs: Additional arguments for SimpleAutoML
    
    Returns:
        SimpleAutoML instance with fitted model
    """
    if learners is None:
        learners = ["random_forest_regressor", "linear_regression", "ridge"]
    
    automl = SimpleAutoML(
        learners=learners,
        task="regression",
        n_trials=n_trials,
        cv=cv,
        **kwargs
    )
    
    automl.fit(X, y)
    return automl


def list_available_learners():
    """
    List all available learners in the registry.
    
    Returns:
        Dictionary with classification and regression learners
    """
    return {
        "classification": get_classification_learners(),
        "regression": get_regression_learners(),
        "all": get_all_learners()
    }


def get_learner_info(learner_name: str):
    """
    Get information about a specific learner.
    
    Args:
        learner_name: Name of the learner
    
    Returns:
        Dictionary with learner information
    """
    try:
        config_space = create_config_space([learner_name])
        return {
            "name": learner_name,
            "config_space": config_space[learner_name],
            "parameters": list(config_space[learner_name].keys()),
            "is_classification": learner_name in get_classification_learners(),
            "is_regression": learner_name in get_regression_learners()
        }
    except ValueError as e:
        return {"error": str(e)} 