# 🔧 Extending MLTeammate

This guide shows you how to extend MLTeammate with custom learners, searchers, callbacks, and other components.

---

## 🎯 Overview

MLTeammate is designed to be highly extensible. You can add:
- **Custom Learners**: Your own machine learning models
- **Custom Searchers**: New hyperparameter optimization strategies
- **Custom Callbacks**: Monitoring and control mechanisms
- **Custom Metrics**: Domain-specific evaluation functions
- **Custom Preprocessors**: Data transformation pipelines

---

## 🧠 Adding Custom Learners

### Basic Learner Template

```python
from sklearn.base import BaseEstimator, ClassifierMixin

class MyCustomLearner(BaseEstimator, ClassifierMixin):
    """
    Custom learner for MLTeammate.
    
    Must implement:
    - __init__(self, config=None, **kwargs)
    - fit(self, X, y)
    - predict(self, X)
    - get_params(self, deep=True)
    - set_params(self, **params)
    """
    
    def __init__(self, config=None, **kwargs):
        """Initialize with configuration."""
        self.config = (config or {}).copy()
        self.config.update(kwargs)
        self.model = None
        
        # Initialize model if config provided
        if self.config:
            self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the underlying model."""
        # Your model initialization logic here
        pass
    
    def fit(self, X, y):
        """Fit the model to the data."""
        if self.model is None:
            self._initialize_model()
        
        # Your training logic here
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities (optional)."""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        raise NotImplementedError("predict_proba not available")
    
    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility."""
        return self.config.copy()
    
    def set_params(self, **params):
        """Set parameters for sklearn compatibility."""
        self.config.update(params)
        return self


# Factory function
def get_my_custom_learner(config):
    """Factory function for the custom learner."""
    return MyCustomLearner(config)
```

### Example: Random Forest Learner

```python
from sklearn.ensemble import RandomForestClassifier

class CustomRandomForestLearner(BaseEstimator, ClassifierMixin):
    def __init__(self, config=None, **kwargs):
        self.config = (config or {}).copy()
        self.config.update(kwargs)
        self.model = None
        
        if self.config:
            self.model = RandomForestClassifier(**self.config)
    
    def fit(self, X, y):
        if self.model is None:
            self.model = RandomForestClassifier(**self.config)
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def get_params(self, deep=True):
        return self.config.copy()
    
    def set_params(self, **params):
        self.config.update(params)
        return self


def get_custom_rf_learner(config):
    return CustomRandomForestLearner(config)
```

### Integration Example

```python
# Define configuration space
config_space = {
    "custom_rf": {
        "n_estimators": {"type": "int", "bounds": [50, 200]},
        "max_depth": {"type": "int", "bounds": [3, 15]},
        "min_samples_split": {"type": "int", "bounds": [2, 10]}
    }
}

# Use in AutoML
learners = {"custom_rf": get_custom_rf_learner}

controller = AutoMLController(
    learners=learners,
    searcher=OptunaSearcher(config_space),
    config_space=config_space,
    task="classification",
    n_trials=10
)
```

---

## 🔍 Adding Custom Searchers

### Basic Searcher Template

```python
class MyCustomSearcher:
    """
    Custom searcher for MLTeammate.
    
    Must implement:
    - __init__(self, config_spaces)
    - suggest(self, trial_id, learner_name)
    - report(self, trial_id, score)
    - get_best(self)
    """
    
    def __init__(self, config_spaces):
        """
        Initialize the searcher.
        
        Args:
            config_spaces: Dictionary of configuration spaces per learner
        """
        self.config_spaces = config_spaces
        self.trials = {}
        self.best_score = None
        self.best_config = None
    
    def suggest(self, trial_id: str, learner_name: str) -> dict:
        """
        Suggest hyperparameters for a trial.
        
        Args:
            trial_id: Unique identifier for the trial
            learner_name: Name of the learner to configure
            
        Returns:
            Dictionary of hyperparameters
        """
        # Your suggestion logic here
        config = {}
        space = self.config_spaces[learner_name]
        
        for name, spec in space.items():
            # Implement your suggestion strategy
            pass
        
        self.trials[trial_id] = config
        return config
    
    def report(self, trial_id: str, score: float):
        """
        Report the score for a trial.
        
        Args:
            trial_id: Trial identifier
            score: Performance score
        """
        if trial_id not in self.trials:
            raise KeyError(f"Unknown trial_id: {trial_id}")
        
        # Update best score tracking
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_config = self.trials[trial_id]
    
    def get_best(self) -> dict:
        """Get the best configuration found so far."""
        return self.best_config or {}
```

### Example: Random Search

```python
import random
import numpy as np

class RandomSearcher:
    def __init__(self, config_spaces, seed=42):
        self.config_spaces = config_spaces
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.trials = {}
        self.best_score = None
        self.best_config = None
    
    def suggest(self, trial_id: str, learner_name: str) -> dict:
        config = {}
        space = self.config_spaces[learner_name]
        
        for name, spec in space.items():
            t = spec["type"]
            if t == "int":
                low, high = spec["bounds"]
                config[name] = random.randint(low, high)
            elif t == "float":
                low, high = spec["bounds"]
                config[name] = random.uniform(low, high)
            elif t == "categorical":
                config[name] = random.choice(spec["choices"])
        
        self.trials[trial_id] = config
        return config
    
    def report(self, trial_id: str, score: float):
        if trial_id not in self.trials:
            raise KeyError(f"Unknown trial_id: {trial_id}")
        
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_config = self.trials[trial_id]
    
    def get_best(self) -> dict:
        return self.best_config or {}
```

### Example: Grid Search

```python
from itertools import product

class GridSearcher:
    def __init__(self, config_spaces):
        self.config_spaces = config_spaces
        self.trials = {}
        self.best_score = None
        self.best_config = None
        self._generate_grid()
    
    def _generate_grid(self):
        """Generate all possible combinations."""
        self.grid_configs = []
        
        for learner_name, space in self.config_spaces.items():
            param_names = list(space.keys())
            param_values = []
            
            for name in param_names:
                spec = space[name]
                if spec["type"] == "categorical":
                    param_values.append(spec["choices"])
                else:
                    # For numeric parameters, create a grid
                    low, high = spec["bounds"]
                    if spec["type"] == "int":
                        values = list(range(low, high + 1))
                    else:
                        # For float, create a small grid
                        values = np.linspace(low, high, 5).tolist()
                    param_values.append(values)
            
            # Generate all combinations
            for combo in product(*param_values):
                config = dict(zip(param_names, combo))
                self.grid_configs.append((learner_name, config))
    
    def suggest(self, trial_id: str, learner_name: str) -> dict:
        if not self.grid_configs:
            raise ValueError("No more configurations to try")
        
        # Get next configuration for this learner
        for i, (name, config) in enumerate(self.grid_configs):
            if name == learner_name:
                self.trials[trial_id] = config
                self.grid_configs.pop(i)
                return config
        
        # If no config for this learner, return empty
        return {}
    
    def report(self, trial_id: str, score: float):
        if trial_id not in self.trials:
            raise KeyError(f"Unknown trial_id: {trial_id}")
        
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_config = self.trials[trial_id]
    
    def get_best(self) -> dict:
        return self.best_config or {}
```

---

## 📊 Adding Custom Callbacks

### Basic Callback Template

```python
class MyCustomCallback:
    """
    Custom callback for MLTeammate.
    
    Optional methods:
    - on_experiment_start(self, experiment_config)
    - on_trial_start(self, trial_id, config)
    - on_trial_end(self, trial_id, config, score, is_best)
    - on_experiment_end(self, best_score, best_config)
    """
    
    def __init__(self, **kwargs):
        """Initialize the callback."""
        self.kwargs = kwargs
    
    def on_experiment_start(self, experiment_config: dict):
        """Called when the experiment starts."""
        print(f"🚀 Experiment started: {experiment_config}")
    
    def on_trial_start(self, trial_id: str, config: dict):
        """Called when a trial starts."""
        print(f"🔬 Trial {trial_id} started with config: {config}")
    
    def on_trial_end(self, trial_id: str, config: dict, score: float, is_best: bool):
        """Called when a trial ends."""
        status = "🏆 BEST" if is_best else "📊"
        print(f"{status} Trial {trial_id} ended - Score: {score:.4f}")
    
    def on_experiment_end(self, best_score: float, best_config: dict):
        """Called when the experiment ends."""
        print(f"🎉 Experiment ended - Best score: {best_score:.4f}")
```

### Example: Early Stopping Callback

```python
class EarlyStoppingCallback:
    def __init__(self, patience=5, min_trials=10):
        self.patience = patience
        self.min_trials = min_trials
        self.trials_since_improvement = 0
        self.best_score = None
        self.should_stop = False
    
    def on_trial_end(self, trial_id: str, config: dict, score: float, is_best: bool):
        if is_best:
            self.best_score = score
            self.trials_since_improvement = 0
        else:
            self.trials_since_improvement += 1
        
        # Check if we should stop
        if (self.trials_since_improvement >= self.patience and 
            hasattr(self, '_trial_count') and self._trial_count >= self.min_trials):
            self.should_stop = True
            print(f"⚠️ Early stopping triggered after {self.patience} trials without improvement")
    
    def on_trial_start(self, trial_id: str, config: dict):
        if not hasattr(self, '_trial_count'):
            self._trial_count = 0
        self._trial_count += 1
```

### Example: Model Saving Callback

```python
import os
import joblib
from datetime import datetime

class ModelSavingCallback:
    def __init__(self, save_dir="./saved_models", save_best_only=True):
        self.save_dir = save_dir
        self.save_best_only = save_best_only
        os.makedirs(save_dir, exist_ok=True)
        self.best_model = None
    
    def on_trial_end(self, trial_id: str, config: dict, score: float, is_best: bool):
        if is_best:
            # Save the best model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"best_model_{timestamp}.pkl"
            filepath = os.path.join(self.save_dir, filename)
            
            # Note: You'll need to access the model from the controller
            # This is a simplified example
            print(f"💾 Best model saved to: {filepath}")
    
    def save_model(self, model, filename=None):
        """Save a model manually."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_{timestamp}.pkl"
        
        filepath = os.path.join(self.save_dir, filename)
        joblib.dump(model, filepath)
        print(f"💾 Model saved to: {filepath}")
```

---

## 📈 Adding Custom Metrics

### Basic Metric Template

```python
def my_custom_metric(y_true, y_pred, **kwargs):
    """
    Custom evaluation metric.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        **kwargs: Additional parameters
        
    Returns:
        float: Metric score
    """
    # Your metric calculation here
    return score
```

### Example: Balanced Accuracy

```python
from sklearn.metrics import balanced_accuracy_score

def balanced_accuracy(y_true, y_pred, **kwargs):
    """Calculate balanced accuracy."""
    return balanced_accuracy_score(y_true, y_pred)
```

### Example: Custom F1 Score

```python
from sklearn.metrics import f1_score

def weighted_f1(y_true, y_pred, **kwargs):
    """Calculate weighted F1 score."""
    return f1_score(y_true, y_pred, average='weighted')
```

### Integration with MLTeammate

```python
# Use custom metric in evaluation
from ml_teammate.utils.metrics import evaluate

# Override the evaluate function or create a custom one
def custom_evaluate(y_true, y_pred, task="classification", metric="custom"):
    if metric == "custom":
        return my_custom_metric(y_true, y_pred)
    else:
        return evaluate(y_true, y_pred, task)
```

---

## 🔄 Adding Custom Preprocessors

### Basic Preprocessor Template

```python
from sklearn.base import BaseEstimator, TransformerMixin

class MyCustomPreprocessor(BaseEstimator, TransformerMixin):
    """
    Custom preprocessor for MLTeammate.
    
    Must implement:
    - fit(self, X, y=None)
    - transform(self, X)
    """
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def fit(self, X, y=None):
        """Fit the preprocessor."""
        # Your fitting logic here
        return self
    
    def transform(self, X):
        """Transform the data."""
        # Your transformation logic here
        return X_transformed
```

### Example: Custom Feature Selector

```python
from sklearn.feature_selection import SelectKBest, f_classif

class CustomFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k=10, score_func=f_classif):
        self.k = k
        self.score_func = score_func
        self.selector = None
    
    def fit(self, X, y=None):
        self.selector = SelectKBest(score_func=self.score_func, k=self.k)
        self.selector.fit(X, y)
        return self
    
    def transform(self, X):
        return self.selector.transform(X)
```

---

## 🧪 Testing Your Extensions

### Unit Tests

```python
import pytest
import numpy as np
from sklearn.datasets import make_classification

def test_custom_learner():
    """Test custom learner functionality."""
    from my_custom_learner import MyCustomLearner
    
    # Create test data
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    
    # Test initialization
    config = {"param1": 10, "param2": 0.5}
    learner = MyCustomLearner(config)
    
    # Test fitting
    learner.fit(X, y)
    
    # Test prediction
    preds = learner.predict(X)
    assert len(preds) == len(y)
    assert preds.dtype in [np.int32, np.int64]
    
    # Test sklearn compatibility
    params = learner.get_params()
    assert "param1" in params
    assert params["param1"] == 10

def test_custom_searcher():
    """Test custom searcher functionality."""
    from my_custom_searcher import MyCustomSearcher
    
    config_spaces = {
        "learner1": {
            "param1": {"type": "int", "bounds": [1, 10]},
            "param2": {"type": "float", "bounds": [0.0, 1.0]}
        }
    }
    
    searcher = MyCustomSearcher(config_spaces)
    
    # Test suggestion
    config = searcher.suggest("trial1", "learner1")
    assert "param1" in config
    assert "param2" in config
    
    # Test reporting
    searcher.report("trial1", 0.8)
    
    # Test best config
    best = searcher.get_best()
    assert best == config
```

### Integration Tests

```python
def test_custom_learner_integration():
    """Test custom learner with AutoMLController."""
    from ml_teammate.automl.controller import AutoMLController
    from ml_teammate.search.optuna_search import OptunaSearcher
    
    # Set up
    learners = {"custom": get_my_custom_learner}
    config_space = {
        "custom": {
            "param1": {"type": "int", "bounds": [1, 10]},
            "param2": {"type": "float", "bounds": [0.0, 1.0]}
        }
    }
    
    controller = AutoMLController(
        learners=learners,
        searcher=OptunaSearcher(config_space),
        config_space=config_space,
        task="classification",
        n_trials=5
    )
    
    # Test data
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    
    # Run AutoML
    controller.fit(X, y)
    
    # Verify results
    assert controller.best_score is not None
    assert controller.best_model is not None
    
    # Test prediction
    preds = controller.predict(X)
    assert len(preds) == len(y)
```

---

## 📚 Best Practices

### 1. Follow Conventions
- Use sklearn-compatible interfaces
- Implement all required methods
- Provide clear documentation
- Use type hints

### 2. Error Handling
- Validate inputs
- Provide meaningful error messages
- Handle edge cases gracefully

### 3. Testing
- Write comprehensive unit tests
- Test edge cases and error conditions
- Verify sklearn compatibility
- Test integration with AutoMLController

### 4. Documentation
- Document all parameters
- Provide usage examples
- Explain the algorithm/approach
- Include performance characteristics

### 5. Performance
- Optimize for speed when possible
- Use efficient data structures
- Consider memory usage
- Profile your code

---

## 🚀 Contributing Back

If you create useful extensions:

1. **Documentation**: Add examples to the docs
2. **Tests**: Include comprehensive test suites
3. **Examples**: Create tutorial examples
4. **Pull Request**: Consider contributing to the main project

---

## 🆘 Getting Help

- **Examples**: Check the tutorials for patterns
- **Issues**: Report bugs or ask questions
- **Discussions**: Share your extensions
- **Code Review**: Get feedback on your implementations

---

## 🎉 Next Steps

1. **Start Simple**: Begin with basic learners
2. **Test Thoroughly**: Ensure your extensions work correctly
3. **Optimize**: Improve performance and usability
4. **Share**: Contribute back to the community

Happy extending! 🚀 