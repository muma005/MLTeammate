"""
MLTeammate Data Layer

Comprehensive data processing capabilities for MLTeammate AutoML framework.
Built on top of our frozen Phase 1 utilities for robust, transparent operation.
"""

# Core data processing classes
from .preprocessing import (
    DataPreprocessor,
    create_simple_preprocessor,
    preprocess_data,
    detect_data_issues
)

from .validation import (
    DataValidator,
    quick_train_test_split,
    quick_data_quality_check
)

from .resampler import (
    DataResampler,
    quick_resample,
    check_imbalance,
    # Legacy compatibility
    oversample_smote,
    undersample_random
)

from .overfit_monitor import (
    OverfitMonitor,
    quick_overfit_check,
    create_early_stopping_monitor
)

# Convenience imports for common workflows
__all__ = [
    # Main classes
    "DataPreprocessor",
    "DataValidator", 
    "DataResampler",
    "OverfitMonitor",
    
    # Quick functions
    "preprocess_data",
    "quick_train_test_split",
    "quick_data_quality_check",
    "quick_resample",
    "check_imbalance",
    "quick_overfit_check",
    
    # Factory functions
    "create_simple_preprocessor",
    "create_early_stopping_monitor",
    
    # Utility functions
    "detect_data_issues",
    
    # Legacy compatibility
    "oversample_smote",
    "undersample_random"
]

# Version info
__version__ = "2.0.0"