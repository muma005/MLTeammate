"""
MLTeammate - Lightweight, Modular AutoML Framework

A transparent and extensible AutoML framework designed for researchers,
students, and developers who want clarity in their ML workflows.

Core Features:
- Simple API requiring no custom code
- 12+ core learners (Random Forest, SVM, XGBoost, LightGBM, etc.)
- Hyperparameter optimization with Optuna/FLAML
- MLflow experiment tracking integration
- Clean modular architecture for research and extension
- Seamless classification and regression support
"""

# Core utilities - always available
from . import utils

# Version information
__version__ = "1.0.0"
__author__ = "MLTeammate Contributors"
__description__ = "Lightweight, modular, transparent AutoML framework"
__url__ = "https://github.com/muma005/MLTeammate"

# Framework metadata
__framework_info__ = {
    "name": "MLTeammate",
    "version": __version__,
    "description": __description__,
    "core_principles": [
        "Transparency - No black-box behavior",
        "Modularity - Clear separation of concerns", 
        "Extensibility - Easy to add new components",
        "Education - Well-documented and understandable"
    ],
    "supported_tasks": ["classification", "regression"],
    "supported_learners": [
        "random_forest", "logistic_regression", "svm", "gradient_boosting",
        "linear_regression", "ridge", "random_forest_regressor", 
        "gradient_boosting_regressor", "xgboost", "lightgbm"
    ],
    "optimization_engines": ["optuna", "flaml"],
    "experiment_tracking": ["mlflow"]
}

# Export framework info for introspection
def get_framework_info():
    """Get MLTeammate framework information."""
    return __framework_info__.copy()

def get_version():
    """Get MLTeammate version."""
    return __version__

# Define what's available at package level
__all__ = [
    "utils",
    "get_framework_info", 
    "get_version",
    "__version__",
    "__framework_info__"
]