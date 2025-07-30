# tutorials/04_add_custom_learner.py
"""
04_add_custom_learner.py
------------------------
Demonstrate how to add custom learners to MLTeammate.
Shows the complete process from creating a custom learner to integrating it with the AutoML pipeline.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin

from ml_teammate.automl.controller import AutoMLController
from ml_teammate.search.optuna_search import OptunaSearcher
from ml_teammate.automl.callbacks import LoggerCallback, ProgressCallback

# ============================================================================
# STEP 1: Create a Custom Learner Class
# ============================================================================

class CustomRandomForestLearner(BaseEstimator, ClassifierMixin):
    """
    Custom Random Forest learner for MLTeammate.
    
    This demonstrates how to create a custom learner that:
    1. Follows sklearn conventions
    2. Accepts configuration dictionaries
    3. Provides proper fit/predict interface
    4. Supports hyperparameter tuning
    """
    
    def __init__(self, config=None, **kwargs):
        """
        Initialize the custom learner.
        
        Args:
            config: Dictionary of hyperparameters
            **kwargs: Additional keyword arguments
        """
        self.config = (config or {}).copy()
        self.config.update(kwargs)
        self.model = None
        
        # Initialize model if config is provided
        if self.config:
            self.model = RandomForestClassifier(**self.config)
    
    def fit(self, X, y):
        """Fit the model to the data."""
        if self.model is None:
            self.model = RandomForestClassifier(**self.config)
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict_proba(X)
    
    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility."""
        return self.config.copy()
    
    def set_params(self, **params):
        """Set parameters for sklearn compatibility."""
        self.config.update(params)
        return self


class CustomLogisticRegressionLearner(BaseEstimator, ClassifierMixin):
    """
    Custom Logistic Regression learner with advanced features.
    
    Demonstrates:
    - Custom preprocessing
    - Feature scaling
    - Advanced configuration handling
    """
    
    def __init__(self, config=None, **kwargs):
        self.config = (config or {}).copy()
        self.config.update(kwargs)
        self.model = None
        self.scaler = None
        
        # Initialize model if config is provided
        if self.config:
            self.model = LogisticRegression(**self.config)
    
    def fit(self, X, y):
        """Fit the model with optional preprocessing."""
        if self.model is None:
            self.model = LogisticRegression(**self.config)
        
        # Optional feature scaling for logistic regression
        if self.config.get('scale_features', False):
            try:
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X)
            except ImportError:
                print("⚠️ sklearn.preprocessing not available. Skipping scaling.")
                X_scaled = X
        else:
            X_scaled = X
        
        self.model.fit(X_scaled, y)
        return self
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.model.predict_proba(X_scaled)
    
    def get_params(self, deep=True):
        return self.config.copy()
    
    def set_params(self, **params):
        self.config.update(params)
        return self


# ============================================================================
# STEP 2: Create Factory Functions
# ============================================================================

def get_custom_rf_learner(config):
    """Factory function for Custom Random Forest learner."""
    return CustomRandomForestLearner(config)

def get_custom_lr_learner(config):
    """Factory function for Custom Logistic Regression learner."""
    return CustomLogisticRegressionLearner(config)


# ============================================================================
# STEP 3: Define Configuration Spaces
# ============================================================================

# Configuration space for Random Forest
custom_rf_config = {
    "n_estimators": {"type": "int", "bounds": [50, 200]},
    "max_depth": {"type": "int", "bounds": [3, 15]},
    "min_samples_split": {"type": "int", "bounds": [2, 10]},
    "min_samples_leaf": {"type": "int", "bounds": [1, 5]},
    "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None]}
}

# Configuration space for Logistic Regression
custom_lr_config = {
    "C": {"type": "float", "bounds": [0.1, 10.0]},
    "penalty": {"type": "categorical", "choices": ["l1", "l2"]},
    "solver": {"type": "categorical", "choices": ["liblinear", "saga"]},
    "scale_features": {"type": "categorical", "choices": [True, False]}
}


# ============================================================================
# STEP 4: Integration Example
# ============================================================================

def run_custom_learner_example():
    """Run a complete example with custom learners."""
    
    print("🔬 Creating synthetic dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"📊 Dataset shape: {X.shape}")
    print(f"🎯 Classes: {np.unique(y)}")
    
    # Define learners and their configurations
    learners = {
        "custom_rf": get_custom_rf_learner,
        "custom_lr": get_custom_lr_learner
    }
    
    config_space = {
        "custom_rf": custom_rf_config,
        "custom_lr": custom_lr_config
    }
    
    # Create callbacks
    callbacks = [
        LoggerCallback(log_level="INFO"),
        ProgressCallback(total_trials=15, patience=5)
    ]
    
    print("🚀 Starting AutoML with custom learners...")
    
    # Create and run AutoML controller
    controller = AutoMLController(
        learners=learners,
        searcher=OptunaSearcher(config_space),
        config_space=config_space,
        task="classification",
        n_trials=15,
        cv=3,
        callbacks=callbacks
    )
    
    # Fit the model
    controller.fit(X_train, y_train)
    
    # Make predictions
    y_pred = controller.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n🎉 Experiment completed!")
    print(f"📈 Best CV Score: {controller.best_score:.4f}")
    print(f"🎯 Test Accuracy: {test_accuracy:.4f}")
    print(f"📊 Best Configuration: {controller.searcher.get_best()}")
    
    return controller, test_accuracy


# ============================================================================
# STEP 5: Advanced Custom Learner Example
# ============================================================================

class EnsembleLearner(BaseEstimator, ClassifierMixin):
    """
    Advanced custom learner that combines multiple models.
    
    Demonstrates:
    - Model composition
    - Custom voting strategies
    - Advanced configuration handling
    """
    
    def __init__(self, config=None, **kwargs):
        self.config = (config or {}).copy()
        self.config.update(kwargs)
        self.models = []
        self.weights = None
        
    def fit(self, X, y):
        """Fit multiple models and determine weights."""
        n_models = self.config.get('n_models', 3)
        model_types = self.config.get('model_types', ['rf', 'lr', 'xgb'])
        
        self.models = []
        
        for i in range(n_models):
            model_type = model_types[i % len(model_types)]
            
            if model_type == 'rf':
                model = RandomForestClassifier(
                    n_estimators=self.config.get('n_estimators', 100),
                    max_depth=self.config.get('max_depth', 10),
                    random_state=42 + i
                )
            elif model_type == 'lr':
                model = LogisticRegression(
                    C=self.config.get('C', 1.0),
                    random_state=42 + i
                )
            else:
                # Default to Random Forest
                model = RandomForestClassifier(random_state=42 + i)
            
            model.fit(X, y)
            self.models.append(model)
        
        # Simple equal weighting
        self.weights = np.ones(len(self.models)) / len(self.models)
        
        return self
    
    def predict(self, X):
        """Make ensemble predictions."""
        if not self.models:
            raise ValueError("Models not fitted. Call fit() first.")
        
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Weighted voting
        weighted_preds = np.zeros(len(X))
        for i, (pred, weight) in enumerate(zip(predictions, self.weights)):
            weighted_preds += weight * pred
        
        return (weighted_preds > 0.5).astype(int)
    
    def get_params(self, deep=True):
        return self.config.copy()
    
    def set_params(self, **params):
        self.config.update(params)
        return self


def get_ensemble_learner(config):
    """Factory function for ensemble learner."""
    return EnsembleLearner(config)


# ============================================================================
# STEP 6: Testing and Validation
# ============================================================================

def test_custom_learners():
    """Test that custom learners work correctly."""
    print("🧪 Testing custom learners...")
    
    # Test data
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    
    # Test Random Forest learner
    rf_learner = get_custom_rf_learner({"n_estimators": 10, "max_depth": 5})
    rf_learner.fit(X, y)
    rf_preds = rf_learner.predict(X)
    print(f"✅ Custom RF learner test passed - Predictions shape: {rf_preds.shape}")
    
    # Test Logistic Regression learner
    lr_learner = get_custom_lr_learner({"C": 1.0, "scale_features": True})
    lr_learner.fit(X, y)
    lr_preds = lr_learner.predict(X)
    print(f"✅ Custom LR learner test passed - Predictions shape: {lr_preds.shape}")
    
    # Test Ensemble learner
    ensemble_learner = get_ensemble_learner({"n_models": 2})
    ensemble_learner.fit(X, y)
    ensemble_preds = ensemble_learner.predict(X)
    print(f"✅ Ensemble learner test passed - Predictions shape: {ensemble_preds.shape}")
    
    print("🎉 All custom learner tests passed!")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("🚀 MLTeammate Custom Learner Tutorial")
    print("=" * 50)
    
    # Test custom learners
    test_custom_learners()
    
    print("\n" + "=" * 50)
    
    # Run full example
    controller, accuracy = run_custom_learner_example()
    
    print(f"\n📚 Tutorial Summary:")
    print(f"   • Created 3 custom learners (RF, LR, Ensemble)")
    print(f"   • Defined configuration spaces for hyperparameter tuning")
    print(f"   • Integrated with MLTeammate AutoML pipeline")
    print(f"   • Achieved test accuracy: {accuracy:.4f}")
    print(f"\n💡 Key Takeaways:")
    print(f"   • Custom learners must follow sklearn conventions")
    print(f"   • Factory functions simplify integration")
    print(f"   • Configuration spaces enable hyperparameter tuning")
    print(f"   • MLTeammate handles the rest automatically!")
