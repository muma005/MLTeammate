"""
Simplified API for MLTeammate

This module provides an extremely user-friendly interface that eliminates the need
for users to write custom classes, functions, or configuration spaces.

Users can simply specify learner names as strings and the framework handles everything else.
"""

import time
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
    
    Pandas-style interface with auto-execution and method chaining.
    """
    
    def __init__(self, 
                 learners: Union[List[str], str] = None,
                 task: str = None,  # Auto-detect if None
                 n_trials: int = None,  # Auto-configure if None
                 cv: Optional[int] = None,  # Auto-configure if None
                 use_mlflow: bool = False,
                 experiment_name: str = None,  # Auto-generate if None
                 log_level: str = "INFO",
                 save_artifacts: bool = True,
                 output_dir: str = "./mlteammate_artifacts"):
        """
        Initialize SimpleAutoML with smart defaults and auto-detection.
        
        Args:
            learners: List of learner names or single learner name as string
                     Examples: ["random_forest", "logistic_regression", "xgboost"]
                              or "random_forest" for single learner
                              Auto-selects if None
            task: "classification" or "regression" (auto-detected if None)
            n_trials: Number of hyperparameter optimization trials (auto-configured if None)
            cv: Number of cross-validation folds (auto-configured if None)
            use_mlflow: Whether to use MLflow for experiment tracking
            experiment_name: Name for MLflow experiment (auto-generated if None)
            log_level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
            save_artifacts: Whether to save models and plots
            output_dir: Directory to save artifacts
        """
        # Store original parameters for method chaining
        self._original_params = {
            'learners': learners,
            'task': task,
            'n_trials': n_trials,
            'cv': cv,
            'use_mlflow': use_mlflow,
            'experiment_name': experiment_name,
            'log_level': log_level,
            'save_artifacts': save_artifacts,
            'output_dir': output_dir
        }
        
        # Initialize with provided parameters
        self.task = task
        self.n_trials = n_trials
        self.cv = cv
        self.use_mlflow = use_mlflow
        self.experiment_name = experiment_name
        self.log_level = log_level
        self.save_artifacts = save_artifacts
        self.output_dir = output_dir
        
        # Method chaining state
        self.searcher_type = "optuna"  # Default searcher
        self.time_budget = None
        self.eci_type = None
        self.eci_params = {}
        
        # Initialize components (will be configured when data is provided)
        self.learner_names = None
        self.learners = None
        self.config_space = None
        self.callbacks = None
        self.controller = None
        self._is_configured = False
    
    def _auto_detect_task(self, y):
        """Auto-detect task type from target variable."""
        import numpy as np
        
        # Check if y is continuous or categorical
        unique_values = np.unique(y)
        n_unique = len(unique_values)
        n_samples = len(y)
        
        # If few unique values relative to sample size, likely classification
        if n_unique <= min(10, n_samples * 0.1):
            return "classification"
        else:
            return "regression"
    
    def _auto_configure_trials(self, X):
        """Auto-configure number of trials based on data size."""
        n_samples, n_features = X.shape
        
        # More trials for larger datasets
        if n_samples > 10000:
            return 20
        elif n_samples > 1000:
            return 15
        elif n_samples > 100:
            return 10
        else:
            return 5
    
    def _auto_configure_cv(self, X):
        """Auto-configure cross-validation folds based on data size."""
        n_samples = X.shape[0]
        
        # More folds for larger datasets, but cap at 5
        if n_samples > 1000:
            return 5
        elif n_samples > 100:
            return 3
        else:
            return 2
    
    def _auto_select_learners(self, task):
        """Auto-select learners based on task."""
        if task == "classification":
            return ["random_forest", "logistic_regression", "xgboost"]
        else:
            return ["random_forest_regressor", "linear_regression", "ridge"]
    
    def _configure_for_data(self, X, y):
        """Configure the AutoML system based on the provided data."""
        if self._is_configured:
            return
        
        # Auto-detect task if not specified
        if self.task is None:
            self.task = self._auto_detect_task(y)
            print(f"🔍 Auto-detected task: {self.task}")
        
        # Auto-configure trials if not specified
        if self.n_trials is None:
            self.n_trials = self._auto_configure_trials(X)
            print(f"⚙️ Auto-configured trials: {self.n_trials}")
        
        # Auto-configure CV if not specified
        if self.cv is None:
            self.cv = self._auto_configure_cv(X)
            print(f"🔄 Auto-configured CV folds: {self.cv}")
        
        # Auto-select learners if not specified
        if self._original_params['learners'] is None:
            self.learner_names = self._auto_select_learners(self.task)
            print(f"🧠 Auto-selected learners: {self.learner_names}")
        else:
            # Handle learners parameter
            learners = self._original_params['learners']
            if isinstance(learners, str):
                learners = [learners]
            self.learner_names = learners
        
        # Auto-generate experiment name if not specified
        if self.experiment_name is None and self.use_mlflow:
            self.experiment_name = f"mlteammate_{self.task}_{int(time.time())}"
            print(f"📊 Auto-generated experiment name: {self.experiment_name}")
        
        # Validate learners
        self._validate_learners(self.learner_names, self.task)
        
        # Create learners and config space automatically
        self.learners = create_learners_dict(self.learner_names)
        self.config_space = create_config_space(self.learner_names)
        
        # Create callbacks
        self.callbacks = self._create_callbacks()
        
        # Initialize controller
        self._create_controller()
        
        self._is_configured = True
    
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
                raise ValueError(f"Learners {incompatible} are not compatible with classification task")
        elif task == "regression":
            regression_learners = get_regression_learners()
            incompatible = [l for l in learners if l not in regression_learners]
            if incompatible:
                raise ValueError(f"Learners {incompatible} are not compatible with regression task")
    
    def _create_callbacks(self):
        """Create callbacks based on configuration."""
        callbacks = []
        
        # Always add progress callback
        callbacks.append(ProgressCallback(total_trials=self.n_trials))
        
        # Add logger callback
        callbacks.append(LoggerCallback(
            use_mlflow=self.use_mlflow,
            log_level=self.log_level,
            experiment_name=self.experiment_name
        ))
        
        # Add artifact callback if requested
        if self.save_artifacts:
            callbacks.append(ArtifactCallback(output_dir=self.output_dir))
        
        return callbacks
    
    def _create_controller(self):
        """Create the AutoML controller."""
        # Create searcher based on configuration
        if self.searcher_type == "flaml":
            from ml_teammate.search import get_searcher
            searcher = get_searcher("flaml", config_spaces=self.config_space, time_budget=self.time_budget)
        else:
            searcher = OptunaSearcher(self.config_space)
        
        self.controller = AutoMLController(
            learners=self.learners,
            searcher=searcher,
            config_space=self.config_space,
            task=self.task,
            n_trials=self.n_trials,
            cv=self.cv,
            callbacks=self.callbacks
        )
    
    # ============================================================================
    # PANDAS-STYLE AUTO-EXECUTION METHODS
    # ============================================================================
    
    def explore_learners(self):
        """Auto-execute learner exploration and print results."""
        print("🔍 Available Learners in MLTeammate")
        print("=" * 50)
        
        learners = list_available_learners()
        
        print("📊 Classification Learners:")
        for i, learner in enumerate(learners["classification"], 1):
            print(f"   {i:2d}. {learner}")
        
        print("\n📈 Regression Learners:")
        for i, learner in enumerate(learners["regression"], 1):
            print(f"   {i:2d}. {learner}")
        
        print(f"\n📋 Total Available Learners: {len(learners['all'])}")
        print("💡 You can use any of these by simply specifying their names as strings!")
        
        return learners
    
    def quick_classify(self, X, y, **kwargs):
        """One-click classification with smart defaults and auto-execution."""
        print("🚀 Running Quick Classification...")
        print("=" * 40)
        
        # Configure for data
        self._configure_for_data(X, y)
        
        # Update with any additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Run the experiment
        self.fit(X, y)
        
        # Print results
        self._print_results()
        
        return self
    
    def quick_regress(self, X, y, **kwargs):
        """One-click regression with smart defaults and auto-execution."""
        print("🚀 Running Quick Regression...")
        print("=" * 40)
        
        # Configure for data
        self._configure_for_data(X, y)
        
        # Update with any additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Run the experiment
        self.fit(X, y)
        
        # Print results
        self._print_results()
        
        return self
    
    def run_example(self, X, y, example_type="auto", **kwargs):
        """Auto-run any example type with smart defaults."""
        if example_type == "auto":
            example_type = self._auto_detect_task(y)
        
        if example_type == "classification":
            return self.quick_classify(X, y, **kwargs)
        elif example_type == "regression":
            return self.quick_regress(X, y, **kwargs)
        else:
            raise ValueError(f"Unknown example type: {example_type}")
    
    def _print_results(self):
        """Auto-print experiment results."""
        print(f"\n🎉 Experiment Results:")
        print(f"📈 Best CV Score: {self.best_score:.4f}")
        print(f"🏆 Best Learner: {self.best_config.get('learner_name', 'unknown')}")
        print(f"⚙️ Best Config: {self.best_config}")
        
        if self.use_mlflow:
            print(f"📊 Results logged to MLflow experiment: {self.experiment_name}")
        
        if self.save_artifacts:
            print(f"💾 Artifacts saved to: {self.output_dir}")
    
    # ============================================================================
    # METHOD CHAINING CONFIGURATION METHODS
    # ============================================================================
    
    def with_mlflow(self, experiment_name=None):
        """Configure MLflow integration and return self for chaining."""
        self.use_mlflow = True
        if experiment_name:
            self.experiment_name = experiment_name
        return self
    
    def with_flaml(self, time_budget=60):
        """Configure FLAML searcher and return self for chaining."""
        self.searcher_type = "flaml"
        self.time_budget = time_budget
        return self
    
    def with_eci(self, eci_type="standard", **kwargs):
        """Configure Early Convergence Indicator and return self for chaining."""
        self.eci_type = eci_type
        self.eci_params = kwargs
        return self
    
    def with_advanced_search(self, searcher_type="optuna", **kwargs):
        """Configure advanced search options and return self for chaining."""
        self.searcher_type = searcher_type
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self
    
    def add_custom_learner(self, name: str, model_class, config_space: Dict[str, Any], **default_params):
        """Add a custom learner without writing def functions."""
        from ml_teammate.learners.registry import get_learner_registry
        from ml_teammate.learners.registry import SklearnWrapper
        
        # Create factory function using lambda (no def needed!)
        factory = lambda config: SklearnWrapper(model_class, {**default_params, **config})
        
        # Register the learner
        registry = get_learner_registry()
        registry._register_learner(name, factory, config_space)
        
        print(f"✅ Custom learner '{name}' registered successfully!")
        return self
    
    def add_ensemble_learner(self, name: str, learners: List[str], config_space: Dict[str, Any]):
        """Add an ensemble learner without writing def functions."""
        from ml_teammate.learners.registry import get_learner_registry
        from sklearn.ensemble import VotingClassifier
        
        def ensemble_factory(config):
            """Factory function for ensemble learner."""
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            
            models = []
            for i, learner_name in enumerate(learners):
                if learner_name == "random_forest":
                    model = RandomForestClassifier(
                        n_estimators=config.get(f'{learner_name}_n_estimators', 100),
                        max_depth=config.get(f'{learner_name}_max_depth', 10),
                        random_state=42 + i
                    )
                elif learner_name == "logistic_regression":
                    model = LogisticRegression(
                        C=config.get(f'{learner_name}_C', 1.0),
                        max_iter=config.get(f'{learner_name}_max_iter', 1000),
                        random_state=42 + i
                    )
                else:
                    # Use default learner
                    from ml_teammate.learners.registry import get_learner
                    model = get_learner(learner_name)(config)
                
                models.append((learner_name, model))
            
            return VotingClassifier(estimators=models, voting='soft')
        
        # Register the ensemble learner
        registry = get_learner_registry()
        registry._register_learner(name, ensemble_factory, config_space)
        
        print(f"✅ Ensemble learner '{name}' registered successfully!")
        return self
    
    # ============================================================================
    # ORIGINAL METHODS (ENHANCED)
    # ============================================================================
    
    def fit(self, X, y):
        """Fit the AutoML model to the data."""
        # Configure for data if not already done
        self._configure_for_data(X, y)
        
        # Recreate controller if configuration changed
        if not self._is_configured:
            self._create_controller()
        
        # Fit the model
        self.controller.fit(X, y)
        return self
    
    def predict(self, X):
        """Make predictions using the best model."""
        if self.controller is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.controller.predict(X)
    
    def score(self, X, y):
        """Score the model on the given data."""
        if self.controller is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.controller.score(X, y)
    
    @property
    def best_score(self):
        """Get the best score achieved during optimization."""
        if self.controller is None:
            return None
        return self.controller.best_score
    
    @property
    def best_model(self):
        """Get the best model found during optimization."""
        if self.controller is None:
            return None
        return self.controller.best_model
    
    @property
    def best_config(self):
        """Get the best configuration found during optimization."""
        if self.controller is None:
            return None
        return self.controller.best_config
    
    def get_results_summary(self):
        """Get a comprehensive summary of the experiment results."""
        if self.controller is None:
            return {}
        
        return {
            "best_score": self.best_score,
            "best_model": type(self.best_model).__name__ if self.best_model else None,
            "best_config": self.best_config,
            "task": self.task,
            "n_trials": self.n_trials,
            "cv": self.cv,
            "learners_used": self.learner_names,
            "experiment_name": self.experiment_name if self.use_mlflow else None
        }


# ============================================================================
# CONVENIENCE FUNCTIONS (ENHANCED)
# ============================================================================

def quick_classification(X, y, 
                        learners: Union[List[str], str] = None,
                        n_trials: int = None,
                        cv: int = None,
                        **kwargs):
    """
    One-liner classification with smart defaults.
    
    Args:
        X: Training features
        y: Training targets
        learners: List of learner names (auto-selected if None)
        n_trials: Number of trials (auto-configured if None)
        cv: Cross-validation folds (auto-configured if None)
        **kwargs: Additional arguments for SimpleAutoML
    
    Returns:
        Fitted SimpleAutoML instance
    """
    automl = SimpleAutoML(
        learners=learners,
        task="classification",
        n_trials=n_trials,
        cv=cv,
        **kwargs
    )
    return automl.quick_classify(X, y)


def quick_regression(X, y,
                    learners: Union[List[str], str] = None,
                    n_trials: int = None,
                    cv: int = None,
                    **kwargs):
    """
    One-liner regression with smart defaults.
    
    Args:
        X: Training features
        y: Training targets
        learners: List of learner names (auto-selected if None)
        n_trials: Number of trials (auto-configured if None)
        cv: Cross-validation folds (auto-configured if None)
        **kwargs: Additional arguments for SimpleAutoML
    
    Returns:
        Fitted SimpleAutoML instance
    """
    automl = SimpleAutoML(
        learners=learners,
        task="regression",
        n_trials=n_trials,
        cv=cv,
        **kwargs
    )
    return automl.quick_regress(X, y)


def list_available_learners():
    """
    Get a dictionary of all available learners organized by task.
    
    Returns:
        Dictionary with keys: "all", "classification", "regression"
    """
    all_learners = get_all_learners()
    classification_learners = get_classification_learners()
    regression_learners = get_regression_learners()
    
    return {
        "all": sorted(all_learners),
        "classification": sorted(classification_learners),
        "regression": sorted(regression_learners)
    }


def get_learner_info(learner_name: str):
    """
    Get detailed information about a specific learner.
    
    Args:
        learner_name: Name of the learner
    
    Returns:
        Dictionary with learner information
    """
    from ml_teammate.learners.registry import get_learner_registry
    
    registry = get_learner_registry()
    
    try:
        # Get the learner factory
        learner_factory = registry.get_learner(learner_name)
        
        # Get the config space
        config_space = registry.get_config_space(learner_name)
        
        # Determine task type
        classification_learners = get_classification_learners()
        task = "classification" if learner_name in classification_learners else "regression"
        
        return {
            "name": learner_name,
            "task": task,
            "config_space": config_space,
            "factory_function": learner_factory.__name__ if hasattr(learner_factory, '__name__') else str(learner_factory)
        }
    except ValueError as e:
        return {"error": str(e)} 