"""
Data Resampling for MLTeammate

Provides data resampling capabilities for handling class imbalance,
including undersampling, oversampling, and synthetic sample generation
using our frozen Phase 1 utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import Counter
try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTEENN
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

# Import our frozen Phase 1 utilities
from ml_teammate.utils import (
    get_logger, validate_data_arrays, ValidationError,
    Timer, time_context
)


class DataResampler:
    """
    Comprehensive data resampling for handling class imbalance in MLTeammate.
    
    Supports various resampling strategies with fallback implementations
    when imbalanced-learn is not available.
    """
    
    def __init__(self, strategy: str = "auto", random_state: int = 42, log_level: str = "INFO"):
        """
        Initialize data resampler.
        
        Args:
            strategy: Resampling strategy ("auto", "oversample", "undersample", "smote", "none")
            random_state: Random seed for reproducibility
            log_level: Logging level
        """
        self.strategy = strategy
        self.random_state = random_state
        self.logger = get_logger(f"DataResampler_{strategy}", log_level)
        
        # Track resampling history
        self.resampling_history: List[Dict[str, Any]] = []
        self.last_resampling_info: Optional[Dict[str, Any]] = None
        
        # Check for imbalanced-learn availability
        if not IMBLEARN_AVAILABLE and strategy in ["smote", "smoteenn"]:
            self.logger.warning(
                f"imbalanced-learn not available, falling back to simple resampling for {strategy}"
            )
        
        self.logger.info(f"DataResampler initialized with strategy: {strategy}")
    
    def analyze_imbalance(self, y) -> Dict[str, Any]:
        """
        Analyze class distribution and imbalance.
        
        Args:
            y: Target vector
            
        Returns:
            dict: Imbalance analysis report
        """
        y = np.asarray(y)
        
        # Get class distribution
        unique_classes, counts = np.unique(y, return_counts=True)
        class_distribution = {str(cls): int(count) for cls, count in zip(unique_classes, counts)}
        
        # Calculate imbalance metrics
        min_count = np.min(counts)
        max_count = np.max(counts)
        total_samples = len(y)
        
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        minority_percentage = (min_count / total_samples) * 100
        
        # Determine imbalance severity
        if imbalance_ratio <= 1.5:
            severity = "balanced"
        elif imbalance_ratio <= 3.0:
            severity = "mild"
        elif imbalance_ratio <= 10.0:
            severity = "moderate"
        else:
            severity = "severe"
        
        analysis = {
            "class_distribution": class_distribution,
            "total_samples": int(total_samples),
            "n_classes": len(unique_classes),
            "imbalance_ratio": float(imbalance_ratio),
            "minority_percentage": float(minority_percentage),
            "majority_class": str(unique_classes[np.argmax(counts)]),
            "minority_class": str(unique_classes[np.argmin(counts)]),
            "severity": severity,
            "needs_resampling": severity in ["moderate", "severe"]
        }
        
        self.logger.info(f"Class imbalance analysis: {severity} (ratio: {imbalance_ratio:.1f}:1)")
        
        return analysis
    
    def _simple_oversample(self, X, y, target_distribution: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple oversampling by duplicating minority class samples.
        
        Args:
            X: Feature matrix
            y: Target vector
            target_distribution: Desired class distribution
            
        Returns:
            tuple: (X_resampled, y_resampled)
        """
        X_resampled = []
        y_resampled = []
        
        for class_label, target_count in target_distribution.items():
            class_indices = np.where(y == class_label)[0]
            current_count = len(class_indices)
            
            if current_count < target_count:
                # Need to oversample this class
                samples_needed = target_count - current_count
                
                # Random sampling with replacement
                rng = np.random.RandomState(self.random_state)
                additional_indices = rng.choice(class_indices, size=samples_needed, replace=True)
                
                # Add original + additional samples
                all_indices = np.concatenate([class_indices, additional_indices])
            else:
                # Use all samples
                all_indices = class_indices
            
            X_resampled.append(X[all_indices])
            y_resampled.append(y[all_indices])
        
        X_resampled = np.vstack(X_resampled)
        y_resampled = np.concatenate(y_resampled)
        
        return X_resampled, y_resampled
    
    def _simple_undersample(self, X, y, target_distribution: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple undersampling by removing majority class samples.
        
        Args:
            X: Feature matrix
            y: Target vector
            target_distribution: Desired class distribution
            
        Returns:
            tuple: (X_resampled, y_resampled)
        """
        X_resampled = []
        y_resampled = []
        
        rng = np.random.RandomState(self.random_state)
        
        for class_label, target_count in target_distribution.items():
            class_indices = np.where(y == class_label)[0]
            current_count = len(class_indices)
            
            if current_count > target_count:
                # Need to undersample this class
                selected_indices = rng.choice(class_indices, size=target_count, replace=False)
            else:
                # Use all samples
                selected_indices = class_indices
            
            X_resampled.append(X[selected_indices])
            y_resampled.append(y[selected_indices])
        
        X_resampled = np.vstack(X_resampled)
        y_resampled = np.concatenate(y_resampled)
        
        return X_resampled, y_resampled
    
    def _determine_resampling_strategy(self, imbalance_analysis: Dict[str, Any]) -> str:
        """
        Automatically determine the best resampling strategy.
        
        Args:
            imbalance_analysis: Results from analyze_imbalance()
            
        Returns:
            str: Recommended strategy
        """
        severity = imbalance_analysis["severity"]
        n_samples = imbalance_analysis["total_samples"]
        minority_percentage = imbalance_analysis["minority_percentage"]
        
        if severity == "balanced":
            return "none"
        elif severity == "mild":
            return "none"  # Usually don't need resampling for mild imbalance
        elif severity == "moderate":
            if n_samples < 1000:
                return "oversample"  # Small dataset, prefer oversampling
            else:
                return "smote" if IMBLEARN_AVAILABLE else "oversample"
        else:  # severe
            if minority_percentage < 5:  # Very few minority samples
                return "smote" if IMBLEARN_AVAILABLE else "oversample"
            elif n_samples > 10000:  # Large dataset
                return "undersample"
            else:
                return "oversample"
    
    def resample(self, X, y, strategy: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample data to handle class imbalance.
        
        Args:
            X: Feature matrix
            y: Target vector
            strategy: Resampling strategy (overrides instance strategy if provided)
            
        Returns:
            tuple: (X_resampled, y_resampled)
        """
        with time_context("DataResampler.resample") as timer:
            # Validate inputs
            X = np.asarray(X)
            y = np.asarray(y)
            validate_data_arrays(X, y, "classification")
            
            # Determine strategy
            if strategy is None:
                strategy = self.strategy
            
            # Analyze current imbalance
            imbalance_analysis = self.analyze_imbalance(y)
            
            # Auto-determine strategy if needed
            if strategy == "auto":
                strategy = self._determine_resampling_strategy(imbalance_analysis)
                self.logger.info(f"Auto-selected resampling strategy: {strategy}")
            
            # Skip resampling if not needed
            if strategy == "none" or not imbalance_analysis["needs_resampling"]:
                self.logger.info("No resampling applied - data is sufficiently balanced")
                return X, y
            
            original_shape = X.shape
            original_distribution = imbalance_analysis["class_distribution"]
            
            self.logger.info(f"Applying {strategy} resampling...")
            self.logger.info(f"Original distribution: {original_distribution}")
            
            # Apply resampling strategy
            if strategy == "oversample":
                X_resampled, y_resampled = self._apply_oversampling(X, y)
            elif strategy == "undersample":
                X_resampled, y_resampled = self._apply_undersampling(X, y)
            elif strategy == "smote":
                X_resampled, y_resampled = self._apply_smote(X, y)
            elif strategy == "smoteenn":
                X_resampled, y_resampled = self._apply_smoteenn(X, y)
            else:
                raise ValidationError(f"Unknown resampling strategy: {strategy}")
            
            # Analyze results
            final_analysis = self.analyze_imbalance(y_resampled)
            final_distribution = final_analysis["class_distribution"]
            
            self.logger.info(f"Final distribution: {final_distribution}")
            self.logger.info(f"Shape: {original_shape} -> {X_resampled.shape}")
            self.logger.info(f"Resampling completed in {timer.get_elapsed():.3f}s")
            
            # Store resampling information
            self.last_resampling_info = {
                "strategy": strategy,
                "original_shape": original_shape,
                "final_shape": X_resampled.shape,
                "original_distribution": original_distribution,
                "final_distribution": final_distribution,
                "original_imbalance_ratio": imbalance_analysis["imbalance_ratio"],
                "final_imbalance_ratio": final_analysis["imbalance_ratio"],
                "timestamp": timer.start_time
            }
            self.resampling_history.append(self.last_resampling_info)
            
        return X_resampled, y_resampled
    
    def _apply_oversampling(self, X, y) -> Tuple[np.ndarray, np.ndarray]:
        """Apply oversampling strategy."""
        if IMBLEARN_AVAILABLE:
            try:
                sampler = RandomOverSampler(random_state=self.random_state)
                return sampler.fit_resample(X, y)
            except Exception as e:
                self.logger.warning(f"imblearn oversampling failed: {e}, using simple method")
        
        # Fallback to simple oversampling
        unique_classes, counts = np.unique(y, return_counts=True)
        max_count = np.max(counts)
        target_distribution = {cls: max_count for cls in unique_classes}
        
        return self._simple_oversample(X, y, target_distribution)
    
    def _apply_undersampling(self, X, y) -> Tuple[np.ndarray, np.ndarray]:
        """Apply undersampling strategy."""
        if IMBLEARN_AVAILABLE:
            try:
                sampler = RandomUnderSampler(random_state=self.random_state)
                return sampler.fit_resample(X, y)
            except Exception as e:
                self.logger.warning(f"imblearn undersampling failed: {e}, using simple method")
        
        # Fallback to simple undersampling
        unique_classes, counts = np.unique(y, return_counts=True)
        min_count = np.min(counts)
        target_distribution = {cls: min_count for cls in unique_classes}
        
        return self._simple_undersample(X, y, target_distribution)
    
    def _apply_smote(self, X, y) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE oversampling."""
        if not IMBLEARN_AVAILABLE:
            self.logger.warning("SMOTE not available, falling back to simple oversampling")
            return self._apply_oversampling(X, y)
        
        try:
            # Check if SMOTE is applicable
            unique_classes, counts = np.unique(y, return_counts=True)
            min_count = np.min(counts)
            
            if min_count < 2:
                self.logger.warning("SMOTE requires at least 2 samples per class, using simple oversampling")
                return self._apply_oversampling(X, y)
            
            if X.shape[1] > 50:
                # For high-dimensional data, use smaller k_neighbors
                k_neighbors = min(5, min_count - 1)
            else:
                k_neighbors = min(5, min_count - 1)
            
            sampler = SMOTE(random_state=self.random_state, k_neighbors=k_neighbors)
            return sampler.fit_resample(X, y)
            
        except Exception as e:
            self.logger.warning(f"SMOTE failed: {e}, falling back to simple oversampling")
            return self._apply_oversampling(X, y)
    
    def _apply_smoteenn(self, X, y) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTEENN combined strategy."""
        if not IMBLEARN_AVAILABLE:
            self.logger.warning("SMOTEENN not available, falling back to SMOTE")
            return self._apply_smote(X, y)
        
        try:
            unique_classes, counts = np.unique(y, return_counts=True)
            min_count = np.min(counts)
            
            if min_count < 2:
                self.logger.warning("SMOTEENN requires at least 2 samples per class, using simple oversampling")
                return self._apply_oversampling(X, y)
            
            k_neighbors = min(5, min_count - 1)
            sampler = SMOTEENN(random_state=self.random_state, 
                              smote=SMOTE(k_neighbors=k_neighbors, random_state=self.random_state))
            return sampler.fit_resample(X, y)
            
        except Exception as e:
            self.logger.warning(f"SMOTEENN failed: {e}, falling back to SMOTE")
            return self._apply_smote(X, y)
    
    def get_resampling_summary(self) -> Dict[str, Any]:
        """
        Get summary of resampling operations.
        
        Returns:
            dict: Resampling history and statistics
        """
        return {
            "default_strategy": self.strategy,
            "random_state": self.random_state,
            "imblearn_available": IMBLEARN_AVAILABLE,
            "last_resampling": self.last_resampling_info,
            "resampling_history": self.resampling_history.copy(),
            "total_operations": len(self.resampling_history)
        }


def quick_resample(X, y, strategy: str = "auto", random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quick data resampling with sensible defaults.
    
    Args:
        X: Feature matrix
        y: Target vector
        strategy: Resampling strategy
        random_state: Random seed
        
    Returns:
        tuple: (X_resampled, y_resampled)
    """
    resampler = DataResampler(strategy=strategy, random_state=random_state)
    return resampler.resample(X, y)


def check_imbalance(y) -> str:
    """
    Quick imbalance check with simple output.
    
    Args:
        y: Target vector
        
    Returns:
        str: Simple imbalance status
    """
    resampler = DataResampler()
    analysis = resampler.analyze_imbalance(y)
    
    severity = analysis["severity"]
    ratio = analysis["imbalance_ratio"]
    
    if severity == "balanced":
        return f"✅ Classes are balanced (ratio: {ratio:.1f}:1)"
    elif severity == "mild":
        return f"✅ Mild imbalance (ratio: {ratio:.1f}:1)"
    elif severity == "moderate":
        return f"⚠️ Moderate imbalance (ratio: {ratio:.1f}:1) - consider resampling"
    else:
        return f"❌ Severe imbalance (ratio: {ratio:.1f}:1) - resampling recommended"


# Legacy functions for backward compatibility
def oversample_smote(X, y):
    """Legacy function for SMOTE oversampling."""
    resampler = DataResampler(strategy="smote")
    return resampler.resample(X, y)


def undersample_random(X, y):
    """Legacy function for random undersampling."""
    resampler = DataResampler(strategy="undersample")
    return resampler.resample(X, y)
