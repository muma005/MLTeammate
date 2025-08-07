# ml_teammate/interface/__init__.py

"""
MLTeammate Interface Module

This module provides user-friendly interfaces for the MLTeammate AutoML framework.
It includes both simple and advanced APIs, as well as command-line interface.
"""

from .simple_api import (
    SimpleAutoML,
    quick_classification,
    quick_regression,
    get_available_learners_by_task,
    get_learner_info
)

from .api import MLTeammate

# Import CLI conditionally to avoid import errors if CLI dependencies are missing
try:
    from .cli import main as cli_main
    __all__ = [
        'SimpleAutoML',
        'MLTeammate',
        'quick_classification',
        'quick_regression',
        'get_available_learners_by_task',
        'get_learner_info',
        'cli_main'
    ]
except ImportError:
    __all__ = [
        'SimpleAutoML',
        'MLTeammate',
        'quick_classification',
        'quick_regression',
        'get_available_learners_by_task',
        'get_learner_info'
    ]

# Version info
__version__ = "1.0.0"
__author__ = "MLTeammate Team"
__email__ = "contact@mlteammate.com"

# Import the simplified API for easy access
from .simple_api import (
    SimpleAutoML,
    quick_classification,
    quick_regression,
    get_available_learners_by_task,
    get_learner_info
)

# Import existing API
from .api import MLTeammate

# Export all interfaces
__all__ = [
    # Simple API (recommended for most users)
    'SimpleAutoML',
    'quick_classification',
    'quick_regression',
    'get_available_learners_by_task',
    'get_learner_info',
    
    # Advanced API
    'MLTeammate'
]
