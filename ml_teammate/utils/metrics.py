"""
Robust Metrics for MLTeammate

Provides reliable evaluation metrics for classification and regression tasks
with comprehensive error handling and validation.
"""

import numpy as np
from typing import Union, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)


def validate_predictions(y_true, y_pred, task: str):
    """
    Validate prediction inputs.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        task: Task type ("classification" or "regression")
        
    Raises:
        ValueError: If inputs are invalid
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true({len(y_true)}) != y_pred({len(y_pred)})")
    
    if len(y_true) == 0:
        raise ValueError("Empty prediction arrays provided")
    
    if task not in ["classification", "regression"]:
        raise ValueError(f"Invalid task: {task}. Use 'classification' or 'regression'")


def evaluate(y_true, y_pred, task: str = "classification") -> float:
    """
    Main evaluation function that returns a single score to maximize.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        task: Task type ("classification" or "regression")
        
    Returns:
        float: Score to maximize (accuracy for classification, negative MSE for regression)
        
    Raises:
        ValueError: If inputs are invalid
    """
    validate_predictions(y_true, y_pred, task)
    
    try:
        if task == "classification":
            return accuracy_score(y_true, y_pred)
        elif task == "regression":
            # Return negative MSE so higher is better (for maximization)
            return -mean_squared_error(y_true, y_pred)
    except Exception as e:
        raise ValueError(f"Error computing {task} metric: {e}")


def classification_metrics(y_true, y_pred, average: str = "weighted") -> dict:
    """
    Comprehensive classification metrics.
    
    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        average: Averaging strategy for multi-class metrics
        
    Returns:
        dict: Dictionary of classification metrics
        
    Raises:
        ValueError: If inputs are invalid
    """
    validate_predictions(y_true, y_pred, "classification")
    
    try:
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
            "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0)
        }
        
        # Add class-specific metrics if multi-class
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        if len(unique_classes) > 2:
            metrics["precision_per_class"] = precision_score(y_true, y_pred, average=None, zero_division=0)
            metrics["recall_per_class"] = recall_score(y_true, y_pred, average=None, zero_division=0)
            metrics["f1_per_class"] = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        return metrics
        
    except Exception as e:
        raise ValueError(f"Error computing classification metrics: {e}")


def regression_metrics(y_true, y_pred) -> dict:
    """
    Comprehensive regression metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        
    Returns:
        dict: Dictionary of regression metrics
        
    Raises:
        ValueError: If inputs are invalid
    """
    validate_predictions(y_true, y_pred, "regression")
    
    try:
        metrics = {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred)
        }
        
        # Add additional useful metrics
        residuals = np.array(y_true) - np.array(y_pred)
        metrics["mean_residual"] = np.mean(residuals)
        metrics["std_residual"] = np.std(residuals)
        
        return metrics
        
    except Exception as e:
        raise ValueError(f"Error computing regression metrics: {e}")


def get_detailed_report(y_true, y_pred, task: str = "classification") -> str:
    """
    Get detailed evaluation report.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        task: Task type ("classification" or "regression")
        
    Returns:
        str: Formatted evaluation report
    """
    validate_predictions(y_true, y_pred, task)
    
    try:
        if task == "classification":
            # Get classification report
            report = classification_report(y_true, y_pred)
            
            # Add confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            report += f"\n\nConfusion Matrix:\n{cm}"
            
            return report
            
        elif task == "regression":
            metrics = regression_metrics(y_true, y_pred)
            
            report = "Regression Metrics:\n"
            report += f"MSE:  {metrics['mse']:.6f}\n"
            report += f"RMSE: {metrics['rmse']:.6f}\n"
            report += f"MAE:  {metrics['mae']:.6f}\n"
            report += f"R²:   {metrics['r2']:.6f}\n"
            report += f"Mean Residual: {metrics['mean_residual']:.6f}\n"
            report += f"Std Residual:  {metrics['std_residual']:.6f}"
            
            return report
            
    except Exception as e:
        return f"Error generating {task} report: {e}"


def calculate_score_improvement(old_score: float, new_score: float) -> dict:
    """
    Calculate improvement metrics between two scores.
    
    Args:
        old_score: Previous score
        new_score: New score
        
    Returns:
        dict: Improvement metrics
    """
    try:
        absolute_improvement = new_score - old_score
        
        if old_score == 0:
            relative_improvement = float('inf') if new_score > 0 else 0
        else:
            relative_improvement = (absolute_improvement / abs(old_score)) * 100
        
        is_improvement = new_score > old_score
        
        return {
            "absolute_improvement": absolute_improvement,
            "relative_improvement_percent": relative_improvement,
            "is_improvement": is_improvement
        }
        
    except Exception as e:
        raise ValueError(f"Error calculating score improvement: {e}")


class MetricCalculator:
    """
    Centralized metric calculation class for consistent evaluation across MLTeammate.
    
    Provides a unified interface for calculating scores, with automatic task detection
    and robust error handling.
    """
    
    def __init__(self):
        """Initialize metric calculator."""
        pass
    
    def calculate_score(self, y_true, y_pred, task: str) -> float:
        """
        Calculate the primary optimization score for a given task.
        
        Args:
            y_true: True target values
            y_pred: Predicted values  
            task: Task type ("classification" or "regression")
            
        Returns:
            float: Score to maximize
        """
        return evaluate(y_true, y_pred, task)
    
    def calculate_detailed_metrics(self, y_true, y_pred, task: str) -> dict:
        """
        Calculate comprehensive metrics for analysis and reporting.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            task: Task type ("classification" or "regression")
            
        Returns:
            dict: Detailed metrics for the task
        """
        if task == "classification":
            return classification_metrics(y_true, y_pred)
        else:
            return regression_metrics(y_true, y_pred)
    
    def get_optimization_direction(self, task: str) -> str:
        """
        Get the optimization direction for a task.
        
        Args:
            task: Task type ("classification" or "regression")
            
        Returns:
            str: "maximize" or "minimize"
        """
        # Both classification (accuracy) and regression (negative MSE) should be maximized
        return "maximize"
