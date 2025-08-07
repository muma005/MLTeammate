"""
Clean Logger for MLTeammate

Pure Python logging without external dependencies.
Provides consistent logging across all MLTeammate components.
"""

import logging
import sys
from typing import Optional
from datetime import datetime


class MLTeammateLogger:
    """
    Clean, lightweight logger for MLTeammate with consistent formatting.
    """
    
    def __init__(self, name: str = "MLTeammate", level: str = "INFO"):
        """
        Initialize logger with specified name and level.
        
        Args:
            name: Logger name (default: "MLTeammate")
            level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        """
        self.logger = logging.getLogger(name)
        self.set_level(level)
        
        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create clean console handler
        self._setup_console_handler()
        
        # Prevent propagation to root logger
        self.logger.propagate = False
    
    def _setup_console_handler(self):
        """Set up clean console output handler."""
        handler = logging.StreamHandler(sys.stdout)
        
        # Clean, readable format
        formatter = logging.Formatter(
            fmt='[%(asctime)s] %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def set_level(self, level: str):
        """
        Set logging level.
        
        Args:
            level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        """
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        
        if level.upper() not in level_map:
            raise ValueError(f"Invalid log level: {level}. Use: {list(level_map.keys())}")
        
        self.logger.setLevel(level_map[level.upper()])
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)


# Global logger instance for convenience
_global_logger: Optional[MLTeammateLogger] = None


def get_logger(name: str = "MLTeammate", level: str = "INFO") -> MLTeammateLogger:
    """
    Get or create a logger instance.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        MLTeammateLogger instance
    """
    global _global_logger
    
    if _global_logger is None:
        _global_logger = MLTeammateLogger(name, level)
    
    return _global_logger


def set_global_log_level(level: str):
    """
    Set global logging level.
    
    Args:
        level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    """
    logger = get_logger()
    logger.set_level(level)


# Convenience functions using global logger
def debug(message: str):
    """Log debug message using global logger."""
    get_logger().debug(message)


def info(message: str):
    """Log info message using global logger."""
    get_logger().info(message)


def warning(message: str):
    """Log warning message using global logger."""
    get_logger().warning(message)


def error(message: str):
    """Log error message using global logger."""
    get_logger().error(message)


def critical(message: str):
    """Log critical message using global logger."""
    get_logger().critical(message)
