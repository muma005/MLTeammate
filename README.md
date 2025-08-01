
# 🤖 MLTeammate

**MLTeammate** is a lightweight, modular AutoML framework built for transparency, experimentation, and extensibility.  
It integrates powerful tools like Optuna, XGBoost, LightGBM, and MLflow — allowing you to build smart pipelines with full control.

> 🔍 Designed for researchers, students, and developers who want clarity in their ML workflows — not just magic.

---

## 🚀 Features

- 🚀 **Simple API** - Use MLTeammate without writing custom code
- 🔄 Cross-validation support (built-in)
- 🧠 **20+ Pre-built learners** (Random Forest, SVM, XGBoost, LightGBM, etc.)
- 🧪 Hyperparameter tuning with Optuna
- 🔬 MLflow experiment tracking (optional)
- ⚙️ Easy to extend with custom learners & config spaces
- 🧩 Clean modular architecture (great for hacking & research)

## 📊 Available Learners

### Classification
- Random Forest, Logistic Regression, SVM, K-Nearest Neighbors
- Decision Trees, Naive Bayes, Linear Discriminant Analysis
- Gradient Boosting, Extra Trees, XGBoost, LightGBM

### Regression  
- Linear Regression, Ridge, Lasso, Random Forest Regressor
- Gradient Boosting Regressor, SVR, K-Nearest Neighbors Regressor

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

### 🚀 Simple API (Recommended for most users)

**No custom code required!** Just specify learner names as strings:

```python
from ml_teammate.interface import SimpleAutoML, quick_classification

# Option 1: SimpleAutoML class
automl = SimpleAutoML(
    learners=["random_forest", "logistic_regression", "xgboost"],
    task="classification",
    n_trials=10,
    cv=3
)
automl.fit(X_train, y_train)
print("Test Score:", automl.score(X_test, y_test))

# Option 2: One-liner function
automl = quick_classification(X_train, y_train, n_trials=10)
print("Best Score:", automl.best_score)
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

## ✅ Alternative Install Options

Once your repo is public on GitHub, users can install it directly using:

```bash
pip install git+https://github.com/yourusername/ml_teammate.git
````

This avoids the need to clone and works well for clean installs.

You can even add a `setup.py` or `pyproject.toml` (optional) to register it on PyPI later if you want `pip install mlteammate`.

---

