# 🚀 Getting Started with MLTeammate

Welcome to MLTeammate! This guide will help you get up and running with your first AutoML experiment.

---

## 🎯 What is MLTeammate?

MLTeammate is a lightweight, modular AutoML framework designed for:
- **Transparency**: No black-box algorithms
- **Extensibility**: Easy to add custom learners and search strategies
- **Research**: Perfect for experimentation and learning
- **Production**: Robust enough for real-world applications

---

## ⚡ Quick Start (5 minutes)

### Step 1: Basic Setup

```python
from ml_teammate.automl.controller import AutoMLController
from ml_teammate.search.optuna_search import OptunaSearcher
from ml_teammate.learners.xgboost_learner import XGBoostLearner
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create synthetic data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Step 2: Define Configuration Space

```python
# Define hyperparameter search space
config_space = {
    "xgboost": {
        "n_estimators": {"type": "int", "bounds": [50, 200]},
        "max_depth": {"type": "int", "bounds": [3, 10]},
        "learning_rate": {"type": "float", "bounds": [0.01, 0.3]}
    }
}
```

### Step 3: Create AutoML Controller

```python
# Set up the AutoML pipeline
controller = AutoMLController(
    learners={"xgboost": XGBoostLearner},
    searcher=OptunaSearcher(config_space),
    config_space=config_space,
    task="classification",
    n_trials=10,
    cv=3
)
```

### Step 4: Run AutoML

```python
# Train the model
controller.fit(X_train, y_train)

# Make predictions
y_pred = controller.predict(X_test)

# Evaluate
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")
```

---

## 📚 Tutorial Series

Follow these tutorials in order for a complete learning experience:

### 1. Basic AutoML
```bash
python ml_teammate/tutorials/01_quickstart_basic.py
```
**Learn**: Basic AutoML workflow, configuration spaces, and evaluation.

### 2. Cross-Validation
```bash
python ml_teammate/tutorials/02_with_cross_validation.py
```
**Learn**: Proper model validation, overfitting prevention, and robust evaluation.

### 3. Experiment Tracking
```bash
python ml_teammate/tutorials/03_with_mlflow.py
```
**Learn**: MLflow integration, artifact management, and experiment reproducibility.

### 4. Custom Learners
```bash
python ml_teammate/tutorials/04_add_custom_learner.py
```
**Learn**: How to add your own machine learning models to the framework.

### 5. Advanced Search
```bash
python ml_teammate/tutorials/05_optuna_search_example.py
```
**Learn**: Advanced hyperparameter optimization techniques and strategies.

---

## 🔧 Core Concepts

### 1. AutoMLController

The main orchestrator that manages the entire AutoML pipeline:

```python
controller = AutoMLController(
    learners=learners,          # Dictionary of learner functions
    searcher=searcher,          # Hyperparameter search strategy
    config_space=config_space,  # Search space definition
    task="classification",      # Task type
    n_trials=10,               # Number of optimization trials
    cv=3                       # Cross-validation folds
)
```

### 2. Learners

Learners are machine learning models wrapped for AutoML:

```python
# Built-in learners
from ml_teammate.learners.xgboost_learner import XGBoostLearner
from ml_teammate.learners.lightgbm_learner import LightGBMLearner

# Custom learners (see tutorial 04)
class MyCustomLearner:
    def __init__(self, config):
        self.config = config
    
    def fit(self, X, y):
        # Training logic
        pass
    
    def predict(self, X):
        # Prediction logic
        pass
```

### 3. Searchers

Searchers handle hyperparameter optimization:

```python
# Optuna searcher (recommended)
from ml_teammate.search.optuna_search import OptunaSearcher

searcher = OptunaSearcher(config_space)

# Custom searchers
class MyCustomSearcher:
    def suggest(self, trial_id, learner_name):
        # Suggest hyperparameters
        pass
    
    def report(self, trial_id, score):
        # Report trial results
        pass
```

### 4. Configuration Spaces

Define the search space for hyperparameters:

```python
config_space = {
    "learner_name": {
        "param1": {"type": "int", "bounds": [1, 100]},
        "param2": {"type": "float", "bounds": [0.0, 1.0]},
        "param3": {"type": "categorical", "choices": ["option1", "option2"]}
    }
}
```

---

## 🎯 Common Use Cases

### Classification

```python
from sklearn.datasets import load_iris

# Load data
X, y = load_iris(return_X_y=True)

# AutoML for classification
controller = AutoMLController(
    learners={"xgboost": XGBoostLearner},
    searcher=OptunaSearcher(config_space),
    config_space=config_space,
    task="classification",
    n_trials=20,
    cv=5
)

controller.fit(X, y)
```

### Regression

```python
from sklearn.datasets import load_boston

# Load data
X, y = load_boston(return_X_y=True)

# AutoML for regression
controller = AutoMLController(
    learners={"xgboost": XGBoostLearner},
    searcher=OptunaSearcher(config_space),
    config_space=config_space,
    task="regression",
    n_trials=20,
    cv=5
)

controller.fit(X, y)
```

### Multiple Learners

```python
# Use multiple learners
learners = {
    "xgboost": XGBoostLearner,
    "lightgbm": LightGBMLearner
}

config_space = {
    "xgboost": xgboost_config,
    "lightgbm": lightgbm_config
}

controller = AutoMLController(
    learners=learners,
    searcher=OptunaSearcher(config_space),
    config_space=config_space,
    task="classification",
    n_trials=30
)
```

---

## 🔍 Advanced Features

### Callbacks

Monitor and control your experiments:

```python
from ml_teammate.automl.callbacks import LoggerCallback, ProgressCallback

callbacks = [
    LoggerCallback(log_level="INFO"),
    ProgressCallback(total_trials=20)
]

controller = AutoMLController(
    # ... other parameters
    callbacks=callbacks
)
```

### MLflow Integration

Track experiments and save artifacts:

```python
from ml_teammate.experiments.mlflow_helper import MLflowHelper

mlflow_helper = MLflowHelper(experiment_name="my_experiment")

controller = AutoMLController(
    # ... other parameters
    mlflow_helper=mlflow_helper
)
```

### Custom Metrics

Define your own evaluation metrics:

```python
from sklearn.metrics import f1_score

def custom_metric(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')

# Use in evaluation
score = custom_metric(y_test, y_pred)
```

---

## 🚨 Best Practices

### 1. Data Preparation
- Always split your data into train/test sets
- Use cross-validation for robust evaluation
- Handle missing values and outliers
- Scale features when appropriate

### 2. Configuration Spaces
- Start with reasonable bounds
- Use log-scale for learning rates
- Include categorical parameters when relevant
- Don't make spaces too large

### 3. Experiment Design
- Start with fewer trials to test setup
- Increase trials for production runs
- Use callbacks for monitoring
- Save and version your experiments

### 4. Model Selection
- Consider multiple learners
- Use domain knowledge to guide search
- Validate on holdout sets
- Monitor for overfitting

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Import Errors
```python
# Make sure you have all dependencies
pip install numpy scikit-learn optuna xgboost
```

#### 2. Memory Issues
```python
# Reduce number of trials or CV folds
controller = AutoMLController(n_trials=5, cv=3)
```

#### 3. Slow Performance
```python
# Use smaller search spaces
# Reduce number of trials
# Use faster learners
```

#### 4. Poor Results
```python
# Check data quality
# Increase number of trials
# Try different learners
# Adjust search space
```

---

## 📖 Next Steps

1. **Complete the Tutorials**: Work through all 5 tutorials
2. **Read the API Documentation**: Understand all available options
3. **Try Your Own Data**: Apply MLTeammate to your datasets
4. **Extend the Framework**: Add custom learners and searchers
5. **Join the Community**: Share your experiences and get help

---

## 🆘 Getting Help

- **Documentation**: Check the docs folder
- **Examples**: Look at the examples folder
- **Issues**: Report bugs on GitHub
- **Discussions**: Ask questions in GitHub Discussions

---

## 🎉 Congratulations!

You've completed the getting started guide! You now have the foundation to use MLTeammate effectively. 

**Next**: Try the tutorials or jump into your own AutoML experiments! 