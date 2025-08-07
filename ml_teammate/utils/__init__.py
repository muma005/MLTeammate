"""
MLTeammate Utilities Package

Provides core utilities for logging, metrics, validation, and performance monitoring.
All utilities are designed to be lightweight, reliable, and dependency-minimal.
"""

# Import core utilities
from .logger import (
    MLTeammateLogger,
    get_logger,
    set_global_log_level,
    debug,
    info,
    warning,
    error,
    critical
)

from .metrics import (
    evaluate,
    classification_metrics,
    regression_metrics,
    get_detailed_report,
    calculate_score_improvement,
    validate_predictions,
    MetricCalculator
)

from .schema import (
    ValidationError,
    validate_config_space,
    validate_learner_config,
    validate_data_arrays,
    validate_trial_config,
    validate_learners_dict,
    validate_cv_folds,
    safe_convert_numeric,
    ensure_numpy_array
)

from .timer import (
    Timer,
    time_context,
    time_function,
    ExperimentTimer,
    format_duration,
    estimate_remaining_time
)

# Define what gets imported with "from ml_teammate.utils import *"
__all__ = [
    # Logging
    "MLTeammateLogger",
    "get_logger",
    "set_global_log_level",
    "debug",
    "info", 
    "warning",
    "error",
    "critical",
    
    # Metrics
    "evaluate",
    "classification_metrics",
    "regression_metrics",
    "get_detailed_report",
    "calculate_score_improvement",
    "validate_predictions",
    "MetricCalculator",
    
    # Schema validation
    "ValidationError",
    "validate_config_space",
    "validate_learner_config",
    "validate_data_arrays", 
    "validate_trial_config",
    "validate_learners_dict",
    "validate_cv_folds",
    "safe_convert_numeric",
    "ensure_numpy_array",
    
    # Timing
    "Timer",
    "time_context",
    "time_function",
    "ExperimentTimer",
    "format_duration",
    "estimate_remaining_time"
]

# Package metadata
__version__ = "1.0.0"
__author__ = "MLTeammate Contributors"
__description__ = "Core utilities for MLTeammate AutoML framework"