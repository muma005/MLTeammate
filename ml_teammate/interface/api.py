
# ml_teammate/interface/api.py
"""
Advanced MLTeammate API Interface

Provides comprehensive AutoML capabilities with full control over the process.
Uses Phase 5 system to ensure compatibility.
"""

from ml_teammate.automl import create_automl_controller
from ml_teammate.automl.callbacks import LoggerCallback, ProgressCallback, MLflowCallback
import pandas as pd
import numpy as np
from typing import Union, Optional, List, Dict, Any


class MLTeammate:
    """
    Advanced MLTeammate API interface providing comprehensive AutoML capabilities.
    
    This interface provides full control over the AutoML process while using the
    Phase 5 system to avoid compatibility issues with frozen phases.
    """
    
    def __init__(self,
                 learners: Optional[List[str]] = None,
                 task: str = "classification",
                 searcher_type: str = "random",
                 n_trials: int = 5,
                 cv_folds: Optional[int] = None,
                 enable_mlflow: bool = False,
                 random_state: int = 42):
        """
        Initialize MLTeammate interface.
        
        Args:
            learners: List of learner names to use (defaults to ["random_forest", "xgboost"])
            task: Type of ML task ("classification" or "regression")
            searcher_type: Search algorithm ("random", "optuna", "flaml")
            n_trials: Number of optimization trials
            cv_folds: Cross-validation folds (None for train/test split)
            enable_mlflow: Whether to enable MLflow tracking
            random_state: Random seed for reproducibility
        """
        self.learners = learners or ["random_forest", "xgboost"]
        self.task = task
        self.searcher_type = searcher_type
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.enable_mlflow = enable_mlflow
        self.random_state = random_state
        
        # Controller will be created when needed
        self.controller = None
        self._is_fitted = False
        
    def _create_controller(self, X, y):
        """Create AutoML controller using Phase 5 system."""
        if self.controller is None:
            # Create callbacks based on configuration
            callbacks = []
            
            # Always add progress callback
            callbacks.append(ProgressCallback(update_interval=5, show_eta=True))
            
            # Add MLflow callback if enabled
            if self.enable_mlflow:
                callbacks.append(MLflowCallback(experiment_name=f"mlteammate_{self.task}"))
            
            # Use Phase 5 create_automl_controller with correct parameters
            self.controller = create_automl_controller(
                learner_names=self.learners,      # ✅ First parameter, correct name
                task=self.task,                   # ✅ Second parameter, correct
                searcher_type=self.searcher_type, # ✅ Correct parameter
                n_trials=self.n_trials,          # ✅ Correct parameter
                cv_folds=self.cv_folds,          # ✅ Correct parameter
                callbacks=callbacks,             # ✅ Correct parameter
                random_state=self.random_state   # ✅ Correct parameter
            )
            
    def fit(self, X: Union[pd.DataFrame, np.ndarray], 
            y: Union[pd.Series, np.ndarray]) -> 'MLTeammate':
        """
        Fit the AutoML system on training data.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            self: Fitted MLTeammate instance
        """
        # Create controller if needed
        self._create_controller(X, y)
        
        # Fit the controller
        self.controller.fit(X, y)
        self._is_fitted = True
        
        return self
        
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features for prediction
            
        Returns:
            Predictions
        """
        if not self._is_fitted:
            raise ValueError("MLTeammate must be fitted before making predictions")
            
        return self.controller.predict(X)
        
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities (classification only).
        
        Args:
            X: Features for prediction
            
        Returns:
            Class probabilities
        """
        if not self._is_fitted:
            raise ValueError("MLTeammate must be fitted before making predictions")
            
        if self.task != "classification":
            raise ValueError("predict_proba is only available for classification tasks")
            
        if hasattr(self.controller, 'predict_proba'):
            return self.controller.predict_proba(X)
        else:
            raise NotImplementedError("predict_proba not implemented for current controller")
            
    def get_best_model(self):
        """Get the best trained model."""
        if not self._is_fitted:
            raise ValueError("MLTeammate must be fitted before accessing best model")
            
        return self.controller.results.get("best_model", None)
    
    @property
    def best_score(self):
        """Get the best score achieved during optimization."""
        if not self._is_fitted:
            raise ValueError("MLTeammate must be fitted before accessing best score")
        
        return self.controller.results.get("best_score", None)
    
    @property  
    def best_config(self):
        """Get the best configuration found during optimization."""
        if not self._is_fitted:
            raise ValueError("MLTeammate must be fitted before accessing best config")
        
        return self.controller.results.get("best_config", None)
    
    @property
    def best_model(self):
        """Get the best model found during optimization."""
        if not self._is_fitted:
            raise ValueError("MLTeammate must be fitted before accessing best model")
        
        return self.controller.results.get("best_model", None)
        
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from the best model."""
        if not self._is_fitted:
            raise ValueError("MLTeammate must be fitted before accessing feature importance")
            
        best_model = self.get_best_model()
        if best_model and hasattr(best_model, 'feature_importances_'):
            # Assuming X was a DataFrame with column names
            return dict(zip(range(len(best_model.feature_importances_)), 
                          best_model.feature_importances_))
        return None
        
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get the optimization history."""
        if not self._is_fitted:
            raise ValueError("MLTeammate must be fitted before accessing optimization history")
            
        return self.controller.results.get("trials", [])
        
    def summary(self) -> Dict[str, Any]:
        """Get a summary of the AutoML run."""
        if not self._is_fitted:
            raise ValueError("MLTeammate must be fitted before generating summary")
            
        return {
            "task": self.task,
            "n_trials": self.n_trials,
            "learners": self.learners,
            "searcher_type": self.searcher_type,
            "cv_folds": self.cv_folds,
            "best_score": self.best_score,
            "best_config": self.best_config,
            "best_model": str(type(self.best_model).__name__) if self.best_model else None
        }
