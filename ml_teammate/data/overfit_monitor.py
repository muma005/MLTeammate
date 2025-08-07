"""
Overfitting Monitor for MLTeammate

Provides overfitting detection and monitoring capabilities
including validation curves, learning curves, and early stopping
using our frozen Phase 1 utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.base import BaseEstimator

# Import our frozen Phase 1 utilities
from ml_teammate.utils import (
    get_logger, validate_data_arrays, ValidationError,
    Timer, time_context, evaluate
)


class OverfitMonitor:
    """
    Comprehensive overfitting detection and monitoring for MLTeammate.
    
    Tracks training vs validation performance, detects overfitting,
    and provides early stopping recommendations.
    """
    
    def __init__(self, task: str = "classification", patience: int = 10, 
                 min_delta: float = 0.001, log_level: str = "INFO"):
        """
        Initialize overfitting monitor.
        
        Args:
            task: Task type ("classification" or "regression")
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            log_level: Logging level
        """
        self.task = task
        self.patience = patience
        self.min_delta = min_delta
        self.logger = get_logger(f"OverfitMonitor_{task}", log_level)
        
        # Monitoring state
        self.training_history: List[Dict[str, float]] = []
        self.best_score: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.epochs_without_improvement: int = 0
        self.is_overfitting: bool = False
        self.early_stop_triggered: bool = False
        
        # Analysis results
        self.overfitting_analysis: Optional[Dict[str, Any]] = None
        self.learning_curve_analysis: Optional[Dict[str, Any]] = None
        
        self.logger.info(f"OverfitMonitor initialized for {task} (patience={patience}, min_delta={min_delta})")
    
    def update(self, train_score: float, val_score: Optional[float] = None, epoch: Optional[int] = None) -> bool:
        """
        Update monitoring with new scores.
        
        Args:
            train_score: Training score for current epoch
            val_score: Validation score for current epoch
            epoch: Current epoch number (auto-incremented if None)
            
        Returns:
            bool: True if should continue training, False if early stop
        """
        if epoch is None:
            epoch = len(self.training_history)
        
        # Store training history
        history_entry = {
            "epoch": epoch,
            "train_score": train_score,
            "val_score": val_score,
            "timestamp": Timer().start_time
        }
        self.training_history.append(history_entry)
        
        # Check for improvement (higher is better for most metrics)
        current_score = val_score if val_score is not None else train_score
        
        if self.best_score is None or current_score > (self.best_score + self.min_delta):
            self.best_score = current_score
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            self.logger.debug(f"New best score: {current_score:.4f} at epoch {epoch}")
        else:
            self.epochs_without_improvement += 1
        
        # Check for overfitting
        if val_score is not None:
            gap = train_score - val_score
            if gap > 0.1:  # 10% gap indicates potential overfitting
                self.is_overfitting = True
        
        # Check early stopping condition
        if self.epochs_without_improvement >= self.patience:
            self.early_stop_triggered = True
            self.logger.info(f"Early stopping triggered at epoch {epoch} (best: {self.best_score:.4f} at epoch {self.best_epoch})")
            return False
        
        return True
    
    def analyze_overfitting(self, window_size: int = 5) -> Dict[str, Any]:
        """
        Analyze training history for overfitting patterns.
        
        Args:
            window_size: Window size for trend analysis
            
        Returns:
            dict: Overfitting analysis report
        """
        if len(self.training_history) < window_size:
            return {"status": "insufficient_data", "message": "Need more training history for analysis"}
        
        # Extract scores
        train_scores = [entry["train_score"] for entry in self.training_history if entry["train_score"] is not None]
        val_scores = [entry["val_score"] for entry in self.training_history if entry["val_score"] is not None]
        
        analysis = {
            "status": "analyzed",
            "total_epochs": len(self.training_history),
            "has_validation_scores": len(val_scores) > 0,
            "overfitting_detected": False,
            "overfitting_severity": "none",
            "recommendations": []
        }
        
        if len(train_scores) > 0:
            analysis["final_train_score"] = float(train_scores[-1])
            analysis["best_train_score"] = float(max(train_scores))
            analysis["train_score_trend"] = self._calculate_trend(train_scores[-window_size:])
        
        if len(val_scores) > 0:
            analysis["final_val_score"] = float(val_scores[-1])
            analysis["best_val_score"] = float(max(val_scores))
            analysis["val_score_trend"] = self._calculate_trend(val_scores[-window_size:])
            
            # Calculate performance gap
            if len(train_scores) > 0 and len(val_scores) > 0:
                final_gap = train_scores[-1] - val_scores[-1]
                max_gap = max([t - v for t, v in zip(train_scores, val_scores) if t is not None and v is not None])
                
                analysis["final_performance_gap"] = float(final_gap)
                analysis["max_performance_gap"] = float(max_gap)
                
                # Determine overfitting severity
                if final_gap > 0.2:  # 20% gap
                    analysis["overfitting_detected"] = True
                    analysis["overfitting_severity"] = "severe"
                    analysis["recommendations"].append("Strong overfitting detected - reduce model complexity")
                elif final_gap > 0.1:  # 10% gap
                    analysis["overfitting_detected"] = True
                    analysis["overfitting_severity"] = "moderate"
                    analysis["recommendations"].append("Moderate overfitting - consider regularization")
                elif final_gap > 0.05:  # 5% gap
                    analysis["overfitting_detected"] = True
                    analysis["overfitting_severity"] = "mild"
                    analysis["recommendations"].append("Mild overfitting - monitor closely")
                
                # Check for diverging trends
                if (analysis.get("train_score_trend", 0) > 0 and 
                    analysis.get("val_score_trend", 0) < -0.01):
                    analysis["overfitting_detected"] = True
                    analysis["recommendations"].append("Training and validation scores are diverging")
        
        # General recommendations
        if analysis["overfitting_detected"]:
            analysis["recommendations"].extend([
                "Consider early stopping",
                "Add regularization (L1/L2)",
                "Reduce model complexity",
                "Increase training data if possible"
            ])
        
        # Check for underfitting
        if len(val_scores) > 0 and val_scores[-1] < 0.7:  # Arbitrary threshold
            analysis["recommendations"].append("Low validation score suggests underfitting - increase model complexity")
        
        self.overfitting_analysis = analysis
        self.logger.info(f"Overfitting analysis: {analysis['overfitting_severity']} overfitting detected")
        
        return analysis
    
    def _calculate_trend(self, scores: List[float]) -> float:
        """
        Calculate trend in scores (slope of linear regression).
        
        Args:
            scores: List of scores
            
        Returns:
            float: Trend slope (positive = improving, negative = declining)
        """
        if len(scores) < 2:
            return 0.0
        
        x = np.arange(len(scores))
        y = np.array(scores)
        
        # Simple linear regression
        n = len(scores)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        return float(slope)
    
    def generate_learning_curve(self, estimator: BaseEstimator, X, y, 
                               train_sizes: Optional[np.ndarray] = None,
                               cv: int = 5) -> Dict[str, Any]:
        """
        Generate learning curve analysis.
        
        Args:
            estimator: Sklearn estimator
            X: Feature matrix
            y: Target vector
            train_sizes: Training set sizes to evaluate
            cv: Cross-validation folds
            
        Returns:
            dict: Learning curve analysis
        """
        with time_context("OverfitMonitor.generate_learning_curve") as timer:
            self.logger.info("Generating learning curve analysis...")
            
            # Validate inputs
            X = np.asarray(X)
            y = np.asarray(y)
            validate_data_arrays(X, y, self.task)
            
            if train_sizes is None:
                train_sizes = np.linspace(0.1, 1.0, 10)
            
            try:
                # Generate learning curve
                train_sizes_abs, train_scores, val_scores = learning_curve(
                    estimator, X, y, train_sizes=train_sizes, cv=cv,
                    random_state=42, n_jobs=-1
                )
                
                # Calculate statistics
                train_mean = np.mean(train_scores, axis=1)
                train_std = np.std(train_scores, axis=1)
                val_mean = np.mean(val_scores, axis=1)
                val_std = np.std(val_scores, axis=1)
                
                analysis = {
                    "status": "success",
                    "train_sizes": train_sizes_abs.tolist(),
                    "train_scores_mean": train_mean.tolist(),
                    "train_scores_std": train_std.tolist(),
                    "val_scores_mean": val_mean.tolist(),
                    "val_scores_std": val_std.tolist(),
                    "final_train_score": float(train_mean[-1]),
                    "final_val_score": float(val_mean[-1]),
                    "final_gap": float(train_mean[-1] - val_mean[-1]),
                    "max_val_score": float(np.max(val_mean)),
                    "convergence_point": None,
                    "recommendations": []
                }
                
                # Analyze convergence
                if len(val_mean) > 3:
                    recent_improvement = val_mean[-1] - val_mean[-3]
                    if abs(recent_improvement) < 0.01:
                        analysis["convergence_point"] = int(train_sizes_abs[-3])
                        analysis["recommendations"].append(f"Learning appears to converge around {analysis['convergence_point']} samples")
                
                # Analyze learning curve shape
                val_trend = self._calculate_trend(val_mean[-5:] if len(val_mean) >= 5 else val_mean)
                if val_trend > 0.01:
                    analysis["recommendations"].append("Validation score still improving - consider more data")
                elif val_trend < -0.01:
                    analysis["recommendations"].append("Validation score declining - possible overfitting")
                else:
                    analysis["recommendations"].append("Validation score stable - good convergence")
                
                # Check final performance gap
                final_gap = analysis["final_gap"]
                if final_gap > 0.1:
                    analysis["recommendations"].append("Large train-validation gap indicates overfitting")
                elif final_gap < 0.02:
                    analysis["recommendations"].append("Small train-validation gap indicates good generalization")
                
                self.learning_curve_analysis = analysis
                self.logger.info(f"Learning curve analysis completed in {timer.get_elapsed():.3f}s")
                
            except Exception as e:
                self.logger.error(f"Learning curve generation failed: {e}")
                analysis = {
                    "status": "failed",
                    "error": str(e),
                    "recommendations": ["Learning curve analysis failed - check data and estimator"]
                }
        
        return analysis
    
    def get_early_stopping_recommendation(self) -> Dict[str, Any]:
        """
        Get early stopping recommendation based on current state.
        
        Returns:
            dict: Early stopping recommendation
        """
        recommendation = {
            "should_stop": self.early_stop_triggered,
            "current_patience": self.epochs_without_improvement,
            "max_patience": self.patience,
            "best_epoch": self.best_epoch,
            "best_score": self.best_score,
            "is_overfitting": self.is_overfitting,
            "recommendation": "continue"
        }
        
        if self.early_stop_triggered:
            recommendation["recommendation"] = "stop_now"
        elif self.epochs_without_improvement > self.patience // 2:
            recommendation["recommendation"] = "stop_soon"
        elif self.is_overfitting:
            recommendation["recommendation"] = "monitor_closely"
        
        return recommendation
    
    def reset(self):
        """Reset monitoring state for new training session."""
        self.training_history.clear()
        self.best_score = None
        self.best_epoch = None
        self.epochs_without_improvement = 0
        self.is_overfitting = False
        self.early_stop_triggered = False
        self.overfitting_analysis = None
        self.learning_curve_analysis = None
        
        self.logger.info("OverfitMonitor reset for new training session")
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive monitoring summary.
        
        Returns:
            dict: Complete monitoring state and analysis
        """
        return {
            "task": self.task,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "training_epochs": len(self.training_history),
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "early_stop_triggered": self.early_stop_triggered,
            "is_overfitting": self.is_overfitting,
            "overfitting_analysis": self.overfitting_analysis,
            "learning_curve_analysis": self.learning_curve_analysis,
            "early_stopping_recommendation": self.get_early_stopping_recommendation()
        }


def quick_overfit_check(train_scores: List[float], val_scores: List[float]) -> str:
    """
    Quick overfitting check with simple output.
    
    Args:
        train_scores: Training scores over time
        val_scores: Validation scores over time
        
    Returns:
        str: Simple overfitting status
    """
    if len(train_scores) == 0 or len(val_scores) == 0:
        return "❓ Insufficient data for overfitting analysis"
    
    final_gap = train_scores[-1] - val_scores[-1]
    
    if final_gap > 0.2:
        return f"❌ Severe overfitting (gap: {final_gap:.1%})"
    elif final_gap > 0.1:
        return f"⚠️ Moderate overfitting (gap: {final_gap:.1%})"
    elif final_gap > 0.05:
        return f"⚠️ Mild overfitting (gap: {final_gap:.1%})"
    else:
        return f"✅ Good generalization (gap: {final_gap:.1%})"


def create_early_stopping_monitor(patience: int = 10, min_delta: float = 0.001, 
                                task: str = "classification") -> OverfitMonitor:
    """
    Create a simple early stopping monitor.
    
    Args:
        patience: Epochs to wait for improvement
        min_delta: Minimum improvement threshold
        task: Task type
        
    Returns:
        OverfitMonitor: Configured monitor
    """
    return OverfitMonitor(task=task, patience=patience, min_delta=min_delta)
