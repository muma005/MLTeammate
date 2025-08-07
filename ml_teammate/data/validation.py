"""
Data Validation for MLTeammate

Provides comprehensive data validation capabilities including cross-validation,
train/test splitting, and data quality checks using our frozen Phase 1 utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Generator
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, KFold,
    cross_val_score, validation_curve
)

# Import our frozen Phase 1 utilities
from ml_teammate.utils import (
    get_logger, validate_data_arrays, ValidationError,
    Timer, time_context
)


class DataValidator:
    """
    Comprehensive data validation and cross-validation for MLTeammate.
    
    Handles train/test splits, cross-validation, and data integrity checks
    with transparent logging and validation.
    """
    
    def __init__(self, task: str = "classification", random_state: int = 42, log_level: str = "INFO"):
        """
        Initialize data validator.
        
        Args:
            task: Task type ("classification" or "regression")
            random_state: Random seed for reproducibility
            log_level: Logging level
        """
        self.task = task
        self.random_state = random_state
        self.logger = get_logger(f"DataValidator_{task}", log_level)
        
        # Validation configuration
        self.test_size = 0.2
        self.cv_folds = 5
        self.validation_history: List[Dict[str, Any]] = []
        
        self.logger.info(f"DataValidator initialized for {task} with random_state={random_state}")
    
    def train_test_split(self, X, y, test_size: Optional[float] = None, 
                        stratify: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Fraction of data for testing (default: 0.2)
            stratify: Whether to stratify split for classification
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        with time_context("DataValidator.train_test_split") as timer:
            # Validate inputs (but allow NaN values for preprocessing workflows)
            X = np.asarray(X)
            y = np.asarray(y)
            # Basic validation without strict NaN check
            if X.shape[0] != y.shape[0]:
                raise ValidationError(f"X and y have different number of samples: {X.shape[0]} vs {y.shape[0]}")
            if self.task == "classification" and len(np.unique(y)) < 2:
                raise ValidationError("Classification requires at least 2 classes")
            if self.task == "regression" and not np.issubdtype(y.dtype, np.number):
                raise ValidationError("Regression targets must be numeric")
            
            if test_size is None:
                test_size = self.test_size
            
            # Determine stratification
            stratify_param = None
            if stratify and self.task == "classification":
                # Check if stratification is possible
                unique_classes, counts = np.unique(y, return_counts=True)
                min_count = np.min(counts)
                if min_count >= 2:  # Need at least 2 samples per class
                    stratify_param = y
                    self.logger.info(f"Using stratified split with {len(unique_classes)} classes")
                else:
                    self.logger.warning("Cannot stratify: some classes have < 2 samples")
            
            # Perform split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state,
                stratify=stratify_param
            )
            
            # Log split information
            self.logger.info(f"Data split completed in {timer.get_elapsed():.3f}s")
            self.logger.info(f"Training set: {X_train.shape[0]} samples")
            self.logger.info(f"Testing set: {X_test.shape[0]} samples")
            self.logger.info(f"Test size: {test_size:.1%}")
            
            # Store validation info
            split_info = {
                "split_type": "train_test_split",
                "test_size": test_size,
                "stratified": stratify_param is not None,
                "train_samples": X_train.shape[0],
                "test_samples": X_test.shape[0],
                "timestamp": timer.start_time
            }
            self.validation_history.append(split_info)
            
        return X_train, X_test, y_train, y_test
    
    def create_cv_folds(self, X, y, n_folds: Optional[int] = None, 
                       stratify: bool = True) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Create cross-validation folds.
        
        Args:
            X: Feature matrix
            y: Target vector
            n_folds: Number of folds (default: 5)
            stratify: Whether to stratify folds for classification
            
        Yields:
            tuple: (train_indices, test_indices) for each fold
        """
        # Validate inputs
        X = np.asarray(X)
        y = np.asarray(y)
        # Basic validation without strict NaN check
        if X.shape[0] != y.shape[0]:
            raise ValidationError(f"X and y have different number of samples: {X.shape[0]} vs {y.shape[0]}")
        if self.task == "classification" and len(np.unique(y)) < 2:
            raise ValidationError("Classification requires at least 2 classes")
        if self.task == "regression" and not np.issubdtype(y.dtype, np.number):
            raise ValidationError("Regression targets must be numeric")
        
        if n_folds is None:
            n_folds = self.cv_folds
        
        # Choose cross-validation strategy
        if stratify and self.task == "classification":
            # Check if stratification is possible
            unique_classes, counts = np.unique(y, return_counts=True)
            min_count = np.min(counts)
            if min_count >= n_folds:
                cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
                self.logger.info(f"Using stratified {n_folds}-fold cross-validation")
            else:
                cv = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
                self.logger.warning(f"Cannot stratify: using regular {n_folds}-fold CV")
        else:
            cv = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
            self.logger.info(f"Using {n_folds}-fold cross-validation")
        
        # Generate folds
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            self.logger.debug(f"Fold {fold_idx + 1}: train={len(train_idx)}, test={len(test_idx)}")
            yield train_idx, test_idx
    
    def validate_data_quality(self, X, y, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Comprehensive data quality validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: Optional feature names
            
        Returns:
            dict: Data quality report with issues and recommendations
        """
        with time_context("DataValidator.validate_data_quality") as timer:
            self.logger.info("Starting comprehensive data quality validation...")
            
            X = np.asarray(X)
            y = np.asarray(y)
            
            quality_report = {
                "overall_status": "unknown",
                "data_shape": {"samples": X.shape[0], "features": X.shape[1]},
                "issues": [],
                "warnings": [],
                "recommendations": [],
                "statistics": {}
            }
            
            # Basic shape validation
            try:
                validate_data_arrays(X, y, self.task)
                quality_report["statistics"]["basic_validation"] = "passed"
            except ValidationError as e:
                quality_report["issues"].append(f"Basic validation failed: {str(e)}")
                quality_report["overall_status"] = "failed"
                return quality_report
            
            # Check for missing values
            missing_count = np.sum(np.isnan(X))
            missing_percentage = missing_count / (X.shape[0] * X.shape[1]) * 100
            quality_report["statistics"]["missing_values"] = {
                "count": int(missing_count),
                "percentage": float(missing_percentage)
            }
            
            if missing_percentage > 20:
                quality_report["issues"].append(f"High missing value percentage: {missing_percentage:.1f}%")
            elif missing_percentage > 5:
                quality_report["warnings"].append(f"Moderate missing values: {missing_percentage:.1f}%")
            
            # Check data types and ranges
            if np.any(np.isinf(X)):
                quality_report["issues"].append("Infinite values detected in features")
            
            # Check feature variance
            low_variance_features = []
            if X.ndim == 2:
                for i in range(X.shape[1]):
                    if np.var(X[:, i]) < 1e-10:
                        feature_name = feature_names[i] if feature_names else f"feature_{i}"
                        low_variance_features.append(feature_name)
                
                if low_variance_features:
                    quality_report["warnings"].append(f"Low variance features: {low_variance_features}")
            
            # Check sample size adequacy
            min_samples_needed = max(100, X.shape[1] * 10)  # Rule of thumb: 10 samples per feature
            if X.shape[0] < min_samples_needed:
                quality_report["warnings"].append(
                    f"Small dataset: {X.shape[0]} samples for {X.shape[1]} features "
                    f"(recommended: {min_samples_needed}+)"
                )
            
            # Classification-specific checks
            if self.task == "classification":
                unique_classes, counts = np.unique(y, return_counts=True)
                quality_report["statistics"]["class_distribution"] = {
                    str(cls): int(count) for cls, count in zip(unique_classes, counts)
                }
                
                # Check class balance
                min_count = np.min(counts)
                max_count = np.max(counts)
                imbalance_ratio = max_count / min_count
                
                if imbalance_ratio > 10:
                    quality_report["issues"].append(
                        f"Severe class imbalance: ratio {imbalance_ratio:.1f}:1"
                    )
                    quality_report["recommendations"].append("Consider resampling techniques")
                elif imbalance_ratio > 3:
                    quality_report["warnings"].append(
                        f"Moderate class imbalance: ratio {imbalance_ratio:.1f}:1"
                    )
                
                # Check minimum samples per class for CV
                if min_count < self.cv_folds:
                    quality_report["warnings"].append(
                        f"Some classes have < {self.cv_folds} samples, may affect cross-validation"
                    )
            
            # Regression-specific checks
            elif self.task == "regression":
                y_std = np.std(y)
                y_range = np.max(y) - np.min(y)
                quality_report["statistics"]["target_statistics"] = {
                    "mean": float(np.mean(y)),
                    "std": float(y_std),
                    "min": float(np.min(y)),
                    "max": float(np.max(y)),
                    "range": float(y_range)
                }
                
                if y_std < 1e-10:
                    quality_report["issues"].append("Target variable has no variance")
            
            # Overall status determination
            if not quality_report["issues"]:
                if not quality_report["warnings"]:
                    quality_report["overall_status"] = "excellent"
                elif len(quality_report["warnings"]) <= 2:
                    quality_report["overall_status"] = "good"
                else:
                    quality_report["overall_status"] = "acceptable"
            else:
                quality_report["overall_status"] = "problematic"
            
            # Add general recommendations
            if missing_count > 0:
                quality_report["recommendations"].append("Consider imputation for missing values")
            
            if X.shape[0] < 1000:
                quality_report["recommendations"].append("Consider gathering more data if possible")
            
            self.logger.info(f"Data quality validation completed in {timer.get_elapsed():.3f}s")
            self.logger.info(f"Overall status: {quality_report['overall_status']}")
            self.logger.info(f"Issues: {len(quality_report['issues'])}, Warnings: {len(quality_report['warnings'])}")
            
        return quality_report
    
    def estimate_cv_time(self, X, y, base_fit_time: float = 1.0, n_folds: Optional[int] = None) -> Dict[str, float]:
        """
        Estimate cross-validation execution time.
        
        Args:
            X: Feature matrix
            y: Target vector
            base_fit_time: Estimated time for single model fit (seconds)
            n_folds: Number of folds
            
        Returns:
            dict: Time estimates for different CV scenarios
        """
        if n_folds is None:
            n_folds = self.cv_folds
        
        n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
        
        # Estimate based on data size (linear scaling assumption)
        size_factor = n_samples / 1000  # Normalize to 1000 samples
        adjusted_fit_time = base_fit_time * size_factor
        
        estimates = {
            "single_fold_time": adjusted_fit_time,
            "total_cv_time": adjusted_fit_time * n_folds,
            "total_cv_time_with_overhead": adjusted_fit_time * n_folds * 1.2,  # 20% overhead
            "estimated_folds": n_folds,
            "data_size_factor": size_factor
        }
        
        self.logger.info(f"CV time estimate: {estimates['total_cv_time']:.1f}s for {n_folds} folds")
        
        return estimates
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get summary of all validation operations performed.
        
        Returns:
            dict: Validation history and statistics
        """
        return {
            "task": self.task,
            "random_state": self.random_state,
            "default_test_size": self.test_size,
            "default_cv_folds": self.cv_folds,
            "validation_history": self.validation_history.copy(),
            "total_operations": len(self.validation_history)
        }


def quick_train_test_split(X, y, test_size: float = 0.2, task: str = "classification", 
                          random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Quick train/test split with sensible defaults.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Fraction for test set
        task: Task type
        random_state: Random seed
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    validator = DataValidator(task=task, random_state=random_state)
    return validator.train_test_split(X, y, test_size=test_size)


def quick_data_quality_check(X, y, task: str = "classification") -> str:
    """
    Quick data quality assessment with simple output.
    
    Args:
        X: Feature matrix
        y: Target vector
        task: Task type
        
    Returns:
        str: Simple quality status
    """
    validator = DataValidator(task=task)
    report = validator.validate_data_quality(X, y)
    
    status = report["overall_status"]
    issue_count = len(report["issues"])
    warning_count = len(report["warnings"])
    
    if status == "excellent":
        return "✅ Data quality is excellent"
    elif status == "good":
        return f"✅ Data quality is good ({warning_count} minor warnings)"
    elif status == "acceptable":
        return f"⚠️ Data quality is acceptable ({warning_count} warnings)"
    else:
        return f"❌ Data quality issues detected ({issue_count} issues, {warning_count} warnings)"
