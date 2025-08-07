"""
MLTeammate AutoML Module

Phase 5: Controller and orchestration system for automated machine learning.

This module provides:
- AutoMLController: Central orchestration with search integration
- Callback system: Flexible experiment monitoring and tracking
- Tuner interface: Unified hyperparameter optimization
- Factory functions: Easy controller and callback creation

Core Components:
- controller.py: Main AutoML orchestration and trial management
- callbacks.py: Experiment lifecycle callbacks and logging
- tuner_interface.py: Unified hyperparameter tuning interface
- Integration with all frozen phases (utilities, data, learners, search)

Usage:
    from ml_teammate.automl import create_automl_controller
    
    # Create controller
    controller = create_automl_controller(
        learner_names=["random_forest_clf", "xgboost_clf"],
        task="classification",
        searcher_type="random",
        n_trials=20
    )
    
    # Fit and predict
    controller.fit(X_train, y_train, X_val, y_val)
    predictions = controller.predict(X_test)
"""

# Import main components
from .controller import AutoMLController, create_automl_controller
from .callbacks import (
    BaseCallback, LoggerCallback, ProgressCallback, 
    MLflowCallback, ArtifactCallback, create_default_callbacks
)
from .tuner_interface import (
    AbstractTuner, SimpleTuner, create_tuner, get_available_tuners
)

# Import convenience functions from automl.py
from .automl import (
    quick_automl, automl_classify, automl_regress
)

# Version and metadata
__version__ = "0.5.0"
__phase__ = "Phase 5: AutoML Controller"

# Public API
__all__ = [
    # Main controller
    "AutoMLController",
    "create_automl_controller",
    
    # Callbacks
    "BaseCallback",
    "LoggerCallback", 
    "ProgressCallback",
    "MLflowCallback",
    "ArtifactCallback",
    "create_default_callbacks",
    
    # Tuner interface
    "AbstractTuner",
    "SimpleTuner", 
    "create_tuner",
    "get_available_tuners",
    
    # Convenience functions
    "quick_automl",
    "automl_classify",
    "automl_regress"
]
