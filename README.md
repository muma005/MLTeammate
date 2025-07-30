
# 🤖 MLTeammate

**MLTeammate** is a lightweight, modular AutoML framework built for transparency, experimentation, and extensibility.  
It integrates powerful tools like Optuna, XGBoost, LightGBM, and MLflow — allowing you to build smart pipelines with full control.

> 🔍 Designed for researchers, students, and developers who want clarity in their ML workflows — not just magic.

---

## 🚀 Features

- 🔄 Cross-validation support (built-in)
- 🧠 Plug-and-play learners (XGBoost, LightGBM, etc.)
- 🧪 Hyperparameter tuning with Optuna
- 🔬 MLflow experiment tracking (optional)
- ⚙️ Easy to extend with custom learners & config spaces
- 🧩 Clean modular architecture (great for hacking & research)

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

## 🧠 Quickstart Example

```python
from ml_teammate.controller import AutoMLController
from ml_teammate.learners.xgboost_learner import get_xgboost_learner
from ml_teammate.search.optuna_search import OptunaSearch
from ml_teammate.utils.config_spaces import xgboost_config_space

controller = AutoMLController(
    learners={"xgboost": get_xgboost_learner},
    config_space={"xgboost": xgboost_config_space},
    searcher=OptunaSearch({"xgboost": xgboost_config_space}),
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

