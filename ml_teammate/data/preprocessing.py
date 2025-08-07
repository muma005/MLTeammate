"""
Data Preprocessing for MLTeammate

Provides clean, robust data preprocessing capabilities including scaling,
encoding, feature selection, and data cleaning using our frozen Phase 1 utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer

# Import our frozen Phase 1 utilities
from ml_teammate.utils import (
    get_logger, validate_data_arrays, ValidationError,
    Timer, time_context
)


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for MLTeammate.
    
    Handles missing values, scaling, encoding, and feature selection
    with transparent logging and validation.
    """
    
    def __init__(self, task: str = "classification", log_level: str = "INFO"):
        """
        Initialize data preprocessor.
        
        Args:
            task: Task type ("classification" or "regression")
            log_level: Logging level
        """
        self.task = task
        self.logger = get_logger(f"DataPreprocessor_{task}", log_level)
        
        # Initialize preprocessing components
        self.scaler: Optional[StandardScaler] = None
        self.imputer: Optional[SimpleImputer] = None
        self.feature_selector: Optional[SelectKBest] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.categorical_encoders: Dict[str, Any] = {}
        
        # Track preprocessing state
        self.is_fitted = False
        self.feature_names: Optional[List[str]] = None
        self.selected_features: Optional[List[int]] = None
        self.preprocessing_stats: Dict[str, Any] = {}
        
        self.logger.info(f"DataPreprocessor initialized for {task}")
    
    def fit(self, X, y=None, feature_names: Optional[List[str]] = None) -> 'DataPreprocessor':
        """
        Fit preprocessing pipeline to training data.
        
        Args:
            X: Feature matrix
            y: Target vector (optional for unsupervised preprocessing)
            feature_names: Optional feature names
            
        Returns:
            self: Fitted preprocessor
        """
        with time_context("DataPreprocessor.fit") as timer:
            self.logger.info("Starting data preprocessing fit...")
            
            # Validate inputs (but allow NaN values since we'll handle them)
            X = np.asarray(X)
            if y is not None:
                y = np.asarray(y)
                # Basic validation without NaN check since we handle missing values
                if X.shape[0] != y.shape[0]:
                    raise ValidationError(f"X and y have different number of samples: {X.shape[0]} vs {y.shape[0]}")
                if self.task == "classification" and len(np.unique(y)) < 2:
                    raise ValidationError("Classification requires at least 2 classes")
                if self.task == "regression" and not np.issubdtype(y.dtype, np.number):
                    raise ValidationError("Regression targets must be numeric")
            
            # Store original shape
            self.preprocessing_stats["original_shape"] = X.shape
            self.preprocessing_stats["original_features"] = X.shape[1]
            
            # Handle feature names
            if feature_names is not None:
                if len(feature_names) != X.shape[1]:
                    raise ValidationError(f"Feature names length ({len(feature_names)}) doesn't match data ({X.shape[1]})")
                self.feature_names = feature_names.copy()
            else:
                self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
            # Check for missing values and handle them
            missing_count = np.sum(np.isnan(X))
            if missing_count > 0:
                self.logger.warning(f"Found {missing_count} missing values, will impute")
                self.imputer = SimpleImputer(strategy='median')
                X = self.imputer.fit_transform(X)
                self.preprocessing_stats["missing_values_imputed"] = missing_count
            
            # Fit scaler
            self.logger.info("Fitting data scaler...")
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            self.preprocessing_stats["scaling_applied"] = True
            
            # Feature selection (if we have targets)
            if y is not None and X.shape[1] > 1:
                n_features = min(X.shape[1], max(10, X.shape[1] // 2))
                self.logger.info(f"Selecting top {n_features} features...")
                
                if self.task == "classification":
                    score_func = f_classif
                else:
                    score_func = f_regression
                
                self.feature_selector = SelectKBest(score_func=score_func, k=n_features)
                X_selected = self.feature_selector.fit_transform(X_scaled, y)
                self.selected_features = self.feature_selector.get_support(indices=True)
                
                self.preprocessing_stats["feature_selection_applied"] = True
                self.preprocessing_stats["features_selected"] = len(self.selected_features)
                self.preprocessing_stats["selected_feature_names"] = [
                    self.feature_names[i] for i in self.selected_features
                ]
            else:
                self.preprocessing_stats["feature_selection_applied"] = False
            
            # Handle categorical target encoding for classification
            if y is not None and self.task == "classification":
                unique_classes = np.unique(y)
                if len(unique_classes) < len(y) and not np.issubdtype(y.dtype, np.integer):
                    self.logger.info("Encoding categorical target labels...")
                    self.label_encoder = LabelEncoder()
                    self.label_encoder.fit(y)
                    self.preprocessing_stats["target_encoding_applied"] = True
                    self.preprocessing_stats["target_classes"] = unique_classes.tolist()
            
            self.is_fitted = True
            self.logger.info(f"Data preprocessing fit completed in {timer.get_elapsed():.3f}s")
            
        return self
    
    def transform(self, X) -> np.ndarray:
        """
        Transform data using fitted preprocessing pipeline.
        
        Args:
            X: Feature matrix to transform
            
        Returns:
            np.ndarray: Transformed feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        with time_context("DataPreprocessor.transform") as timer:
            X = np.asarray(X)
            
            # Validate shape
            if X.shape[1] != self.preprocessing_stats["original_features"]:
                raise ValidationError(
                    f"Feature count mismatch: expected {self.preprocessing_stats['original_features']}, "
                    f"got {X.shape[1]}"
                )
            
            # Apply imputation if fitted
            if self.imputer is not None:
                X = self.imputer.transform(X)
            
            # Apply scaling
            if self.scaler is not None:
                X = self.scaler.transform(X)
            
            # Apply feature selection
            if self.feature_selector is not None:
                X = self.feature_selector.transform(X)
            
            self.logger.debug(f"Data transformed in {timer.get_elapsed():.3f}s")
            
        return X
    
    def fit_transform(self, X, y=None, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Fit and transform data in one step.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: Optional feature names
            
        Returns:
            np.ndarray: Transformed feature matrix
        """
        return self.fit(X, y, feature_names).transform(X)
    
    def transform_target(self, y) -> np.ndarray:
        """
        Transform target labels if label encoder was fitted.
        
        Args:
            y: Target vector
            
        Returns:
            np.ndarray: Transformed target vector
        """
        if self.label_encoder is not None:
            return self.label_encoder.transform(y)
        return np.asarray(y)
    
    def inverse_transform_target(self, y) -> np.ndarray:
        """
        Inverse transform target labels.
        
        Args:
            y: Encoded target vector
            
        Returns:
            np.ndarray: Original target labels
        """
        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform(y)
        return np.asarray(y)
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of features after preprocessing.
        
        Returns:
            List[str]: Feature names after selection
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        if self.selected_features is not None:
            return [self.feature_names[i] for i in self.selected_features]
        else:
            return self.feature_names.copy()
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive preprocessing summary.
        
        Returns:
            dict: Preprocessing statistics and configuration
        """
        if not self.is_fitted:
            return {"status": "not_fitted"}
        
        summary = self.preprocessing_stats.copy()
        summary.update({
            "status": "fitted",
            "task": self.task,
            "final_feature_count": len(self.get_feature_names()),
            "final_feature_names": self.get_feature_names()
        })
        
        return summary


def create_simple_preprocessor(task: str = "classification") -> DataPreprocessor:
    """
    Create a simple preprocessor with default settings.
    
    Args:
        task: Task type ("classification" or "regression")
        
    Returns:
        DataPreprocessor: Configured preprocessor
    """
    return DataPreprocessor(task=task)


def preprocess_data(X, y=None, task: str = "classification", 
                   feature_names: Optional[List[str]] = None) -> Tuple[np.ndarray, Optional[np.ndarray], DataPreprocessor]:
    """
    Convenience function for quick data preprocessing.
    
    Args:
        X: Feature matrix
        y: Target vector (optional)
        task: Task type
        feature_names: Optional feature names
        
    Returns:
        tuple: (X_processed, y_processed, preprocessor)
    """
    logger = get_logger("preprocess_data")
    
    with time_context("preprocess_data") as timer:
        preprocessor = DataPreprocessor(task=task)
        X_processed = preprocessor.fit_transform(X, y, feature_names)
        
        y_processed = None
        if y is not None:
            y_processed = preprocessor.transform_target(y)
        
        logger.info(f"Data preprocessing completed in {timer.get_elapsed():.3f}s")
        logger.info(f"Shape: {X.shape} -> {X_processed.shape}")
        
    return X_processed, y_processed, preprocessor


def detect_data_issues(X, y=None, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Detect potential issues in the dataset.
    
    Args:
        X: Feature matrix
        y: Target vector (optional)
        feature_names: Optional feature names
        
    Returns:
        dict: Data quality report
    """
    logger = get_logger("detect_data_issues")
    
    X = np.asarray(X)
    if y is not None:
        y = np.asarray(y)
    
    issues = {
        "missing_values": {},
        "constant_features": [],
        "high_cardinality_features": [],
        "outliers": {},
        "class_imbalance": None
    }
    
    # Check missing values
    if X.ndim == 2:
        for i in range(X.shape[1]):
            missing_count = np.sum(np.isnan(X[:, i]))
            if missing_count > 0:
                feature_name = feature_names[i] if feature_names else f"feature_{i}"
                issues["missing_values"][feature_name] = {
                    "count": int(missing_count),
                    "percentage": float(missing_count / len(X) * 100)
                }
    
    # Check constant features
    if X.ndim == 2:
        for i in range(X.shape[1]):
            if np.var(X[:, i]) < 1e-10:  # Nearly constant
                feature_name = feature_names[i] if feature_names else f"feature_{i}"
                issues["constant_features"].append(feature_name)
    
    # Check class imbalance for classification
    if y is not None:
        unique_values, counts = np.unique(y, return_counts=True)
        if len(unique_values) > 1:
            min_count = np.min(counts)
            max_count = np.max(counts)
            imbalance_ratio = max_count / min_count
            
            issues["class_imbalance"] = {
                "ratio": float(imbalance_ratio),
                "class_distribution": {str(val): int(count) for val, count in zip(unique_values, counts)},
                "is_imbalanced": imbalance_ratio > 5.0
            }
    
    # Log findings
    total_issues = (
        len(issues["missing_values"]) + 
        len(issues["constant_features"]) +
        (1 if issues["class_imbalance"] and issues["class_imbalance"]["is_imbalanced"] else 0)
    )
    
    if total_issues > 0:
        logger.warning(f"Detected {total_issues} potential data quality issues")
    else:
        logger.info("No major data quality issues detected")
    
    return issues
