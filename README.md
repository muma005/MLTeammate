
# 🤖 MLTeammate

**Your AI-Powered Machine Learning Teammate** — Automate the tedious, focus on the insights.

> Transform 200+ lines of manual ML code into 3 lines of intelligent automation 🚀

---

## 🔥 **Why MLTeammate?**

### **😤 Traditional ML Workflow (The Hard Way)**

```python
# 1. Manual data preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np

# 2. Split data manually
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Define multiple models manually
models = {
    'rf': RandomForestClassifier(random_state=42),
    'lr': LogisticRegression(random_state=42),
    'svm': SVC(random_state=42)
}

# 4. Define hyperparameter grids manually
param_grids = {
    'rf': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
    'lr': {
        'C': [0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs']
    },
    'svm': {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear']
    }
}

# 5. Manual hyperparameter tuning for each model
best_models = {}
for name, model in models.items():
    print(f"Tuning {name}...")
    grid_search = GridSearchCV(
        model, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_models[name] = grid_search.best_estimator_
    print(f"{name} best params: {grid_search.best_params_}")
    print(f"{name} best CV score: {grid_search.best_score_:.4f}")

# 6. Evaluate each model manually
results = {}
for name, model in best_models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"{name} Test Accuracy: {accuracy:.4f}")

# 7. Find best model manually
best_model_name = max(results, key=results.get)
best_model = best_models[best_model_name]
print(f"Best Model: {best_model_name} with accuracy: {results[best_model_name]:.4f}")

# 8. Manual experiment tracking (if you remember)
# ... more manual logging code ...

# Total: 50+ lines of repetitive code, manual tuning, no experiment tracking
```

**Problems with Traditional Approach:**
- ❌ **50+ lines** of repetitive boilerplate code
- ❌ **Manual hyperparameter** grid definition (error-prone)
- ❌ **No intelligent search** (just grid search)
- ❌ **No experiment tracking** (lose your results)
- ❌ **No cross-validation** integration
- ❌ **Time-consuming** setup for every project
- ❌ **Easy to make mistakes** in model comparison

---

### 🚀 **MLTeammate Workflow (The Smart Way)**

```python
from ml_teammate.interface.api import MLTeammate

# That's it! MLTeammate handles everything intelligently:
automl = MLTeammate(
    learners=["random_forest", "logistic_regression", "svm"],
    task="classification"
)
automl.fit(X_train, y_train)
predictions = automl.predict(X_test)

print(f"Best Model: {automl.best_config['learner_name']}")
print(f"Test Accuracy: {automl.score(X_test, y_test):.4f}")
```

**✅ MLTeammate Automatically Handles:**
- ✅ **Intelligent hyperparameter optimization** (Optuna/FLAML)
- ✅ **Cross-validation** with proper scoring
- ✅ **Model comparison** and selection
- ✅ **Experiment tracking** (MLflow integration)
- ✅ **Smart search spaces** for each algorithm
- ✅ **Performance monitoring** and logging
- ✅ **Best model selection** based on CV scores
- ✅ **Error handling** and validation

---

## 📊 **Real Impact: Before vs After**

| Aspect | Traditional ML | MLTeammate | **Improvement** |
|--------|----------------|------------|-----------------|
| **Lines of Code** | 50+ lines | 3 lines | **94% reduction** |
| **Setup Time** | 30-60 minutes | 30 seconds | **99% faster** |
| **Hyperparameter Tuning** | Manual grids | Intelligent search | **10x smarter** |
| **Model Comparison** | Manual tracking | Automatic | **Error-free** |
| **Experiment Tracking** | Manual/None | Auto MLflow | **Built-in** |
| **Cross-Validation** | Manual setup | Automatic | **Zero config** |
| **Best Practices** | Hope you remember | Always applied | **Guaranteed** |

---

## 🚀 Features

- 🚀 **Simple API** - Use MLTeammate without writing custom code
- 🔄 Cross-validation support (built-in)
- 🧠 **12 Core learners** (Random Forest, SVM, XGBoost, LightGBM, etc.)
- 🧪 Hyperparameter tuning with Optuna
- 🔬 MLflow experiment tracking (optional)
- ⚙️ Easy to extend with custom learners & config spaces
- 🧩 Clean modular architecture (great for hacking & research)

## 📊 Available Learners

### Classification (6)
- Random Forest, Logistic Regression, SVM, Gradient Boosting
- XGBoost, LightGBM

### Regression (6)  
- Linear Regression, Ridge, Random Forest Regressor
- Gradient Boosting Regressor, XGBoost, LightGBM

**Just specify learner names as strings!** No custom classes needed.

---

## 📦 Installation

You can install MLTeammate in two ways:

### Option 1: Clone the repo (recommended for development)

```bash
git clone https://github.com/yourusername/ml_teammate.git
cd ml_teammate
pip install -r requirements.txt
````

### Option 2: Install directly via pip (⚠️ only works if  published )

```bash
pip install git+https://github.com/yourusername/ml_teammate.git
```

> You can also install it in editable mode for development:

```bash
pip install -e .
```

---

## 🧠 Quickstart Examples

### 🚀 Pandas-Style API (Recommended for most users)

**Zero function definitions required!** Just like using pandas:

```python
from ml_teammate.interface import SimpleAutoML, quick_classification

# Option 1: Auto-execution with smart defaults
automl = SimpleAutoML()
automl.quick_classify(X_train, y_train)  # Auto-executes and prints results!

# Option 2: Method chaining (like pandas)
automl = SimpleAutoML()
automl.with_mlflow().with_flaml(time_budget=30).quick_classify(X_train, y_train)

# Option 3: One-liner function (ultimate simplicity)
automl = quick_classification(X_train, y_train)  # Auto-configures everything!

# Option 4: Explore available learners
automl = SimpleAutoML()
automl.explore_learners()  # Auto-prints all available learners
```

### 🔧 Advanced API (For power users)

```python
from ml_teammate.automl.controller import AutoMLController
from ml_teammate.learners.xgboost_learner import get_xgboost_learner
from ml_teammate.search.optuna_search import OptunaSearcher
from ml_teammate.search.config_space import xgboost_config

controller = AutoMLController(
    learners={"xgboost": get_xgboost_learner},
    config_space={"xgboost": xgboost_config},
    searcher=OptunaSearcher({"xgboost": xgboost_config}),
    task="classification",
    n_trials=5,
    cv=3
)

controller.fit(X_train, y_train)
print("Test Score:", controller.score(X_test, y_test))
```

More examples in [`/tutorials`](./tutorials)

---

## 📂 Project Structure

```
ml_teammate/
├── controller/           # Core logic
├── learners/             # XGBoost, LightGBM, etc.
├── search/               # Optuna, FLAML, etc.
├── utils/                # Metrics, config spaces
├── tutorials/            # Usage examples
├── docs/                 # Modular documentation
└── README.md
```

---

## 🧩 Extending MLTeammate

* Add a custom learner to `learners/`
* Define its config space in `utils/config_spaces.py`
* Register it in your controller + search space

See [`04_add_custom_learner.py`](./tutorials/04_add_custom_learner.py) for a working example.

---

## 📜 License

MIT License — open for improvement and contribution.

---

## 🙌 Author

Built  by a learner [Muma005](https://github.com/muma005)
Feel free to fork, star ⭐, and share feedback.

````

---

### ✅ **Direct GitHub Installation**

you can install MLTeammate directly from GitHub using:

```bash
pip install git+https://github.com/muma005/MLTeammate.git
```

**This works immediately and includes:**
- ✅ All latest features and fixes
- ✅ No need to clone the repository  
- ✅ Automatic dependency installation
- ✅ Works in any Python environment

### **Alternative Installation Methods:**

```bash
# Install from specific branch
pip install git+https://github.com/muma005/MLTeammate.git@main

# Install in development mode (for contributors)
git clone https://github.com/muma005/MLTeammate.git
cd MLTeammate
pip install -e .

# Future PyPI installation (coming soon!)
pip install ml-teammate
```

---


---

