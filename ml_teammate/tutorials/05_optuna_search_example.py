# tutorials/05_optuna_search_example.py
"""
05_optuna_search_example.py
---------------------------
Demonstrate advanced Optuna search capabilities in MLTeammate.
Shows different search strategies, pruning, and custom optimization objectives.
"""

import numpy as np
import optuna
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, make_scorer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ml_teammate.automl.controller import AutoMLController
from ml_teammate.search.optuna_search import OptunaSearcher
from ml_teammate.learners.xgboost_learner import XGBoostLearner
from ml_teammate.automl.callbacks import LoggerCallback, ProgressCallback

# ============================================================================
# EXAMPLE 1: Basic Optuna Search with Different Samplers
# ============================================================================

def demonstrate_samplers():
    """Demonstrate different Optuna samplers."""
    print("🔬 Example 1: Different Optuna Samplers")
    print("-" * 50)
    
    # Create dataset
    X, y = make_classification(n_samples=500, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Configuration space
    config_space = {
        "xgboost": {
            "n_estimators": {"type": "int", "bounds": [50, 200]},
            "max_depth": {"type": "int", "bounds": [3, 10]},
            "learning_rate": {"type": "float", "bounds": [0.01, 0.3]},
            "subsample": {"type": "float", "bounds": [0.6, 1.0]}
        }
    }
    
    # Test different samplers
    samplers = {
        "TPE": optuna.samplers.TPESampler(seed=42),
        "Random": optuna.samplers.RandomSampler(seed=42),
        "CmaEs": optuna.samplers.CmaEsSampler(seed=42),
        "NSGAII": optuna.samplers.NSGAIISampler(seed=42)
    }
    
    results = {}
    
    for sampler_name, sampler in samplers.items():
        print(f"\n🧪 Testing {sampler_name} sampler...")
        
        # Create custom searcher with specific sampler
        study = optuna.create_study(direction="maximize", sampler=sampler)
        custom_searcher = OptunaSearcher(config_space, study=study)
        
        # Run AutoML
        controller = AutoMLController(
            learners={"xgboost": XGBoostLearner},
            searcher=custom_searcher,
            config_space=config_space,
            task="classification",
            n_trials=10,
            cv=3,
            callbacks=[LoggerCallback(log_level="WARNING")]
        )
        
        controller.fit(X_train, y_train)
        test_score = controller.score(X_test, y_test)
        
        results[sampler_name] = {
            "best_cv_score": controller.best_score,
            "test_score": test_score,
            "best_config": controller.searcher.get_best()
        }
        
        print(f"   Best CV Score: {controller.best_score:.4f}")
        print(f"   Test Score: {test_score:.4f}")
    
    # Compare results
    print(f"\n📊 Sampler Comparison:")
    print(f"{'Sampler':<10} {'CV Score':<10} {'Test Score':<10}")
    print("-" * 30)
    for sampler, result in results.items():
        print(f"{sampler:<10} {result['best_cv_score']:<10.4f} {result['test_score']:<10.4f}")
    
    return results


# ============================================================================
# EXAMPLE 2: Multi-Objective Optimization
# ============================================================================

class MultiObjectiveSearcher(OptunaSearcher):
    """Custom searcher for multi-objective optimization."""
    
    def __init__(self, config_spaces, objectives=["accuracy", "speed"]):
        self.objectives = objectives
        self.study = optuna.create_study(
            directions=["maximize", "minimize"],  # maximize accuracy, minimize time
            sampler=optuna.samplers.NSGAIISampler(seed=42)
        )
        self.config_spaces = config_spaces
        self.trials = {}
    
    def suggest(self, trial_id: str, learner_name: str) -> dict:
        trial = self.study.ask()
        self.trials[trial_id] = trial
        
        space = self.config_spaces[learner_name]
        config = {}
        for name, spec in space.items():
            t = spec["type"]
            if t == "int":
                low, high = spec["bounds"]
                config[name] = trial.suggest_int(name, low, high)
            elif t == "float":
                low, high = spec["bounds"]
                config[name] = trial.suggest_float(name, low, high)
            elif t == "categorical":
                config[name] = trial.suggest_categorical(name, spec["choices"])
        return config
    
    def report(self, trial_id: str, scores: list):
        """Report multiple objectives."""
        trial = self.trials.get(trial_id)
        if not trial:
            raise KeyError(f"Unknown trial_id: {trial_id}")
        self.study.tell(trial, scores)
    
    def get_best(self) -> dict:
        """Get best configuration from Pareto front."""
        if self.study.best_trials:
            return self.study.best_trials[0].params
        return {}


def demonstrate_multi_objective():
    """Demonstrate multi-objective optimization."""
    print("\n🔬 Example 2: Multi-Objective Optimization")
    print("-" * 50)
    
    # Create dataset
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Configuration space
    config_space = {
        "xgboost": {
            "n_estimators": {"type": "int", "bounds": [50, 300]},
            "max_depth": {"type": "int", "bounds": [3, 15]},
            "learning_rate": {"type": "float", "bounds": [0.01, 0.5]},
            "subsample": {"type": "float", "bounds": [0.5, 1.0]}
        }
    }
    
    # Custom controller for multi-objective
    class MultiObjectiveController(AutoMLController):
        def fit(self, X, y):
            experiment_config = {
                "task": self.task,
                "n_trials": self.n_trials,
                "cv": self.cv,
                "learners": list(self.learners.keys()),
                "data_shape": X.shape
            }
            
            for cb in self.callbacks:
                cb.on_experiment_start(experiment_config)
            
            for i in range(self.n_trials):
                trial_id = str(uuid.uuid4())
                start_time = time.time()
                learner_name = "xgboost"
                
                try:
                    config = self.searcher.suggest(trial_id, learner_name)
                    
                    for cb in self.callbacks:
                        cb.on_trial_start(trial_id, config)
                    
                    model = self.learners[learner_name](config)
                    
                    # Measure training time
                    train_start = time.time()
                    if self.cv:
                        cv_scores = cross_val_score(model, X, y, cv=self.cv, scoring='accuracy')
                        model.fit(X, y)
                    else:
                        model.fit(X, y)
                        cv_scores = [model.score(X, y)]
                    
                    train_time = time.time() - train_start
                    
                    # Calculate objectives
                    accuracy = np.mean(cv_scores)
                    speed = train_time  # in seconds
                    
                    # Report both objectives
                    self.searcher.report(trial_id, [accuracy, speed])
                    
                    # Track best accuracy model
                    is_best = self.best_score is None or accuracy > self.best_score
                    if is_best:
                        self.best_score = accuracy
                        self.best_model = model
                    
                    for cb in self.callbacks:
                        cb.on_trial_end(trial_id, config, accuracy, is_best)
                    
                except Exception as e:
                    print(f"[Warning] Trial {i+1} failed: {e}")
                    continue
            
            for cb in self.callbacks:
                cb.on_experiment_end(self.best_score, self.searcher.get_best())
    
    # Run multi-objective optimization
    searcher = MultiObjectiveSearcher(config_space)
    controller = MultiObjectiveController(
        learners={"xgboost": XGBoostLearner},
        searcher=searcher,
        config_space=config_space,
        task="classification",
        n_trials=20,
        cv=3,
        callbacks=[ProgressCallback(total_trials=20)]
    )
    
    controller.fit(X_train, y_train)
    
    # Analyze Pareto front
    print(f"\n📊 Pareto Front Analysis:")
    print(f"Number of Pareto optimal solutions: {len(searcher.study.best_trials)}")
    
    for i, trial in enumerate(searcher.study.best_trials[:5]):  # Show top 5
        print(f"Solution {i+1}: Accuracy={trial.values[0]:.4f}, Time={trial.values[1]:.2f}s")
        print(f"  Config: {trial.params}")
    
    return controller


# ============================================================================
# EXAMPLE 3: Custom Optimization Objective
# ============================================================================

def custom_accuracy_scorer(y_true, y_pred):
    """Custom accuracy scorer with class balance penalty."""
    accuracy = accuracy_score(y_true, y_pred)
    
    # Penalty for class imbalance
    unique, counts = np.unique(y_pred, return_counts=True)
    if len(counts) > 1:
        balance_penalty = 1 - (min(counts) / max(counts))
    else:
        balance_penalty = 1.0
    
    # Combine accuracy and balance
    balanced_score = accuracy * (1 - 0.1 * balance_penalty)
    return balanced_score


def demonstrate_custom_objective():
    """Demonstrate custom optimization objective."""
    print("\n🔬 Example 3: Custom Optimization Objective")
    print("-" * 50)
    
    # Create imbalanced dataset
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_classes=2,
        weights=[0.8, 0.2],  # Imbalanced classes
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Class distribution: {np.bincount(y)}")
    
    # Configuration space
    config_space = {
        "xgboost": {
            "n_estimators": {"type": "int", "bounds": [50, 200]},
            "max_depth": {"type": "int", "bounds": [3, 10]},
            "learning_rate": {"type": "float", "bounds": [0.01, 0.3]},
            "scale_pos_weight": {"type": "float", "bounds": [0.5, 5.0]}
        }
    }
    
    # Custom controller with custom objective
    class CustomObjectiveController(AutoMLController):
        def fit(self, X, y):
            experiment_config = {
                "task": self.task,
                "n_trials": self.n_trials,
                "cv": self.cv,
                "learners": list(self.learners.keys()),
                "data_shape": X.shape
            }
            
            for cb in self.callbacks:
                cb.on_experiment_start(experiment_config)
            
            for i in range(self.n_trials):
                trial_id = str(uuid.uuid4())
                start_time = time.time()
                learner_name = "xgboost"
                
                try:
                    config = self.searcher.suggest(trial_id, learner_name)
                    
                    for cb in self.callbacks:
                        cb.on_trial_start(trial_id, config)
                    
                    model = self.learners[learner_name](config)
                    
                    if self.cv:
                        preds = cross_val_predict(model, X, y, cv=self.cv)
                        model.fit(X, y)
                    else:
                        model.fit(X, y)
                        preds = model.predict(X)
                    
                    # Use custom objective
                    custom_score = custom_accuracy_scorer(y, preds)
                    
                    self.searcher.report(trial_id, custom_score)
                    
                    is_best = self.best_score is None or custom_score > self.best_score
                    if is_best:
                        self.best_score = custom_score
                        self.best_model = model
                    
                    for cb in self.callbacks:
                        cb.on_trial_end(trial_id, config, custom_score, is_best)
                    
                except Exception as e:
                    print(f"[Warning] Trial {i+1} failed: {e}")
                    continue
            
            for cb in self.callbacks:
                cb.on_experiment_end(self.best_score, self.searcher.get_best())
    
    # Run custom objective optimization
    controller = CustomObjectiveController(
        learners={"xgboost": XGBoostLearner},
        searcher=OptunaSearcher(config_space),
        config_space=config_space,
        task="classification",
        n_trials=15,
        cv=3,
        callbacks=[ProgressCallback(total_trials=15)]
    )
    
    controller.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = controller.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_custom_score = custom_accuracy_scorer(y_test, y_pred)
    
    print(f"\n📊 Results:")
    print(f"Best Custom Score: {controller.best_score:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Custom Score: {test_custom_score:.4f}")
    print(f"Best Config: {controller.searcher.get_best()}")
    
    return controller


# ============================================================================
# EXAMPLE 4: Regression with Optuna
# ============================================================================

def demonstrate_regression():
    """Demonstrate Optuna search for regression tasks."""
    print("\n🔬 Example 4: Regression with Optuna")
    print("-" * 50)
    
    # Create regression dataset
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Configuration space for regression
    config_space = {
        "xgboost": {
            "n_estimators": {"type": "int", "bounds": [50, 300]},
            "max_depth": {"type": "int", "bounds": [3, 12]},
            "learning_rate": {"type": "float", "bounds": [0.01, 0.3]},
            "subsample": {"type": "float", "bounds": [0.6, 1.0]},
            "colsample_bytree": {"type": "float", "bounds": [0.6, 1.0]}
        }
    }
    
    # Custom regression learner
    class XGBoostRegressorLearner:
        def __init__(self, config=None, **kwargs):
            self.config = (config or {}).copy()
            self.config.update(kwargs)
            self.model = None
            
            if self.config:
                import xgboost as xgb
                self.model = xgb.XGBRegressor(**self.config)
        
        def fit(self, X, y):
            if self.model is None:
                import xgboost as xgb
                self.model = xgb.XGBRegressor(**self.config)
            self.model.fit(X, y)
            return self
        
        def predict(self, X):
            return self.model.predict(X)
        
        def get_params(self, deep=True):
            return self.config.copy()
        
        def set_params(self, **params):
            self.config.update(params)
            return self
    
    def get_xgboost_regressor_learner(config):
        return XGBoostRegressorLearner(config)
    
    # Run regression optimization
    controller = AutoMLController(
        learners={"xgboost": get_xgboost_regressor_learner},
        searcher=OptunaSearcher(config_space),
        config_space=config_space,
        task="regression",
        n_trials=15,
        cv=3,
        callbacks=[ProgressCallback(total_trials=15)]
    )
    
    controller.fit(X_train, y_train)
    
    # Evaluate
    y_pred = controller.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    
    print(f"\n📊 Regression Results:")
    print(f"Best CV Score: {controller.best_score:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test RMSE: {np.sqrt(test_mse):.4f}")
    print(f"Best Config: {controller.searcher.get_best()}")
    
    return controller


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import time
    import uuid
    
    print("🚀 MLTeammate Optuna Search Examples")
    print("=" * 60)
    
    # Run all examples
    results = {}
    
    # Example 1: Different samplers
    results["samplers"] = demonstrate_samplers()
    
    # Example 2: Multi-objective optimization
    results["multi_objective"] = demonstrate_multi_objective()
    
    # Example 3: Custom objective
    results["custom_objective"] = demonstrate_custom_objective()
    
    # Example 4: Regression
    results["regression"] = demonstrate_regression()
    
    print(f"\n🎉 All examples completed!")
    print(f"\n📚 Tutorial Summary:")
    print(f"   • Tested 4 different Optuna samplers")
    print(f"   • Implemented multi-objective optimization")
    print(f"   • Created custom optimization objectives")
    print(f"   • Applied Optuna to regression tasks")
    print(f"\n💡 Key Takeaways:")
    print(f"   • Optuna provides flexible search strategies")
    print(f"   • Multi-objective optimization finds Pareto optimal solutions")
    print(f"   • Custom objectives enable domain-specific optimization")
    print(f"   • MLTeammate integrates seamlessly with Optuna's advanced features")
