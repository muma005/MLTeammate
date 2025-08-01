# tutorials/05_optuna_search_example.py
"""
05_optuna_search_example.py
---------------------------
Demonstrate advanced Optuna search capabilities in MLTeammate with pandas-style interface.

This tutorial showcases:
1. Different Optuna samplers (TPE, Random, CmaEs, NSGAII)
2. Multi-objective optimization
3. Custom optimization objectives
4. Regression with Optuna
5. Advanced search strategies

Perfect for users who want to explore advanced optimization techniques!
"""

import numpy as np
import optuna
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, make_scorer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Import the pandas-style API
from ml_teammate.interface import SimpleAutoML
from ml_teammate.search.optuna_search import OptunaSearcher
from ml_teammate.learners.registry import get_learner_registry

# ============================================================================
# STEP 1: Explore Available Searchers (No Functions!)
# ============================================================================

print("🔍 STEP 1: Explore Available Searchers")
print("=" * 50)

# Import search components
from ml_teammate.search import list_available_searchers

# List available searchers
searchers = list_available_searchers()
print("📊 Available Searchers:")
for name, info in searchers.items():
    print(f"   • {name}: {info['description']}")
    print(f"     Features: {', '.join(info['features'])}")
    print(f"     Dependencies: {', '.join(info['dependencies'])}")
    print()

# ============================================================================
# STEP 2: Different Optuna Samplers (No Functions!)
# ============================================================================

print("\n🔬 STEP 2: Different Optuna Samplers")
print("=" * 50)

# Generate sample data
X, y = make_classification(n_samples=500, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Dataset shape: {X.shape}")
print(f"🎯 Classes: {np.unique(y)}")

# Test different samplers using SimpleAutoML
samplers_to_test = ["TPE", "Random", "CmaEs", "NSGAII"]
results = {}

for sampler_name in samplers_to_test:
    print(f"\n🧪 Testing {sampler_name} sampler:")
    
    try:
        # Use SimpleAutoML with specific sampler
        automl = SimpleAutoML(
            learners=["random_forest", "logistic_regression"],
            task="classification",
            n_trials=5,
            cv=3
        )
        
        # Configure with specific sampler
        automl.with_advanced_search(searcher_type="optuna", sampler=sampler_name).quick_classify(X_train, y_train)
        
        # Test the model
        y_pred = automl.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        results[sampler_name] = {
            "best_cv_score": automl.best_score,
            "test_accuracy": test_accuracy,
            "best_config": automl.best_config
        }
        
        print(f"   ✅ Best CV Score: {automl.best_score:.4f}")
        print(f"   🎯 Test Accuracy: {test_accuracy:.4f}")
        print(f"   🏆 Best Learner: {automl.best_config.get('learner_name', 'unknown')}")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        results[sampler_name] = {"error": str(e)}

# Compare results
print(f"\n📊 Sampler Comparison:")
print(f"{'Sampler':<10} {'CV Score':<10} {'Test Score':<10}")
print("-" * 30)
for sampler, result in results.items():
    if "error" not in result:
        print(f"{sampler:<10} {result['best_cv_score']:<10.4f} {result['test_accuracy']:<10.4f}")
    else:
        print(f"{sampler:<10} {'Failed':<10} {'N/A':<10}")

# ============================================================================
# STEP 3: Multi-Objective Optimization (No Functions!)
# ============================================================================

print("\n🎯 STEP 3: Multi-Objective Optimization")
print("=" * 50)

# Generate data
X, y = make_classification(n_samples=600, n_features=15, n_informative=8, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Dataset shape: {X.shape}")

# Method 1: Using SimpleAutoML with multi-objective
print("\n🔬 Method 1: Multi-objective with SimpleAutoML")
automl = SimpleAutoML(
    learners=["random_forest", "logistic_regression"],
    task="classification",
    n_trials=5,
    cv=3
)

# Configure for multi-objective optimization
automl.with_advanced_search(
    searcher_type="optuna", 
    sampler="NSGAII",
    objectives=["accuracy", "speed"]
).quick_classify(X_train, y_train)

# Test the model
y_pred = automl.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Test Accuracy: {test_accuracy:.4f}")

# Method 2: Direct multi-objective searcher
print("\n🔬 Method 2: Direct multi-objective searcher")
try:
    # Create custom multi-objective study
    study = optuna.create_study(
        directions=["maximize", "minimize"],  # maximize accuracy, minimize time
        sampler=optuna.samplers.NSGAIISampler(seed=42)
    )
    
    # Create custom searcher
    config_space = {
        "random_forest": {
            "n_estimators": {"type": "int", "bounds": [50, 200]},
            "max_depth": {"type": "int", "bounds": [3, 10]}
        },
        "logistic_regression": {
            "C": {"type": "float", "bounds": [0.1, 10.0]},
            "max_iter": {"type": "int", "bounds": [100, 1000]}
        }
    }
    
    multi_searcher = OptunaSearcher(config_space, study=study)
    
    # Use with SimpleAutoML
    automl = SimpleAutoML(
        learners=["random_forest", "logistic_regression"],
        task="classification",
        n_trials=5,
        cv=3
    )
    
    # Override searcher
    automl.controller.searcher = multi_searcher
    automl.quick_classify(X_train, y_train)
    
    print("✅ Multi-objective optimization completed!")
    
except Exception as e:
    print(f"❌ Multi-objective failed: {e}")

# ============================================================================
# STEP 4: Custom Optimization Objectives (No Functions!)
# ============================================================================

print("\n⚡ STEP 4: Custom Optimization Objectives")
print("=" * 50)

# Generate data
X, y = make_classification(n_samples=500, n_features=12, n_informative=6, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Dataset shape: {X.shape}")

# Method 1: Using SimpleAutoML with custom objective
print("\n🔬 Method 1: Custom objective with SimpleAutoML")

# Define custom objective function
def custom_balanced_accuracy(y_true, y_pred):
    """Custom balanced accuracy with class balance penalty."""
    from sklearn.metrics import balanced_accuracy_score
    from collections import Counter
    
    # Calculate balanced accuracy
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Add penalty for class imbalance
    class_counts = Counter(y_true)
    imbalance_penalty = 1 - (min(class_counts.values()) / max(class_counts.values()))
    
    return bal_acc - 0.1 * imbalance_penalty

# Use SimpleAutoML with custom scoring
automl = SimpleAutoML(
    learners=["random_forest", "logistic_regression"],
    task="classification",
    n_trials=5,
    cv=3
)

# Configure with custom objective
automl.with_advanced_search(
    searcher_type="optuna",
    sampler="TPE",
    custom_scorer=custom_balanced_accuracy
).quick_classify(X_train, y_train)

# Test the model
y_pred = automl.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Test Accuracy: {test_accuracy:.4f}")

# Method 2: Direct custom objective
print("\n🔬 Method 2: Direct custom objective")
try:
    # Create custom study with custom objective
    def objective(trial):
        # Suggest hyperparameters
        learner_name = trial.suggest_categorical("learner_name", ["random_forest", "logistic_regression"])
        
        if learner_name == "random_forest":
            n_estimators = trial.suggest_int("n_estimators", 50, 200)
            max_depth = trial.suggest_int("max_depth", 3, 10)
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        else:
            C = trial.suggest_float("C", 0.1, 10.0)
            max_iter = trial.suggest_int("max_iter", 100, 1000)
            model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
        
        # Evaluate with custom scoring
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring=make_scorer(custom_balanced_accuracy))
        return scores.mean()
    
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=5)
    
    print(f"✅ Custom objective optimization completed!")
    print(f"🏆 Best custom score: {study.best_value:.4f}")
    print(f"⚙️ Best params: {study.best_params}")
    
except Exception as e:
    print(f"❌ Custom objective failed: {e}")

# ============================================================================
# STEP 5: Regression with Optuna (No Functions!)
# ============================================================================

print("\n📈 STEP 5: Regression with Optuna")
print("=" * 50)

# Generate regression data
X_reg, y_reg = make_regression(n_samples=500, n_features=15, n_informative=8, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

print(f"📊 Dataset shape: {X_reg.shape}")

# Method 1: SimpleAutoML for regression
print("\n🔬 Method 1: SimpleAutoML for regression")
automl_reg = SimpleAutoML(
    learners=["random_forest_regressor", "linear_regression", "ridge"],
    task="regression",
    n_trials=5,
    cv=3
)

automl_reg.with_advanced_search(searcher_type="optuna", sampler="TPE").quick_regress(X_train_reg, y_train_reg)

# Test the model
y_pred_reg = automl_reg.predict(X_test_reg)
test_r2 = r2_score(y_test_reg, y_pred_reg)
print(f"🎯 Test R² Score: {test_r2:.4f}")

# Method 2: Direct regression optimization
print("\n🔬 Method 2: Direct regression optimization")
try:
    def regression_objective(trial):
        # Suggest hyperparameters for Random Forest Regressor
        n_estimators = trial.suggest_int("n_estimators", 50, 200)
        max_depth = trial.suggest_int("max_depth", 3, 15)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        
        # Evaluate with R² scoring
        scores = cross_val_score(model, X_train_reg, y_train_reg, cv=3, scoring='r2')
        return scores.mean()
    
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(regression_objective, n_trials=5)
    
    print(f"✅ Regression optimization completed!")
    print(f"🏆 Best R² score: {study.best_value:.4f}")
    print(f"⚙️ Best params: {study.best_params}")
    
except Exception as e:
    print(f"❌ Regression optimization failed: {e}")

# ============================================================================
# STEP 6: Advanced Search Strategies (No Functions!)
# ============================================================================

print("\n🚀 STEP 6: Advanced Search Strategies")
print("=" * 50)

# Generate data
X, y = make_classification(n_samples=400, n_features=10, n_informative=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Dataset shape: {X.shape}")

# Strategy 1: Pruning with SimpleAutoML
print("\n🔬 Strategy 1: Pruning with SimpleAutoML")
automl = SimpleAutoML(
    learners=["random_forest", "logistic_regression"],
    task="classification",
    n_trials=8,
    cv=3
)

automl.with_advanced_search(
    searcher_type="optuna",
    sampler="TPE",
    pruner="MedianPruner"
).quick_classify(X_train, y_train)

# Test the model
y_pred = automl.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Test Accuracy: {test_accuracy:.4f}")

# Strategy 2: Early stopping
print("\n🔬 Strategy 2: Early stopping")
automl = SimpleAutoML(
    learners=["random_forest", "logistic_regression"],
    task="classification",
    n_trials=10,
    cv=3
)

automl.with_advanced_search(
    searcher_type="optuna",
    sampler="TPE",
    early_stopping=True,
    patience=3
).quick_classify(X_train, y_train)

# Test the model
y_pred = automl.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Test Accuracy: {test_accuracy:.4f}")

# ============================================================================
# STEP 7: Method Chaining with Optuna (No Functions!)
# ============================================================================

print("\n🔗 STEP 7: Method Chaining with Optuna")
print("=" * 50)

# Generate data
X, y = make_classification(n_samples=300, n_features=8, n_informative=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Dataset shape: {X.shape}")

# Chain multiple Optuna configurations
print("\n🔬 Method chaining with Optuna:")

# Example 1: TPE + Pruning + MLflow
print("\n🔬 Example 1: TPE + Pruning + MLflow")
automl = SimpleAutoML()
automl.with_advanced_search(
    searcher_type="optuna",
    sampler="TPE",
    pruner="MedianPruner"
).with_mlflow(experiment_name="optuna_advanced").quick_classify(X_train, y_train)

# Example 2: NSGAII + Multi-objective + Custom scoring
print("\n🔬 Example 2: NSGAII + Multi-objective + Custom scoring")
automl = SimpleAutoML()
automl.with_advanced_search(
    searcher_type="optuna",
    sampler="NSGAII",
    objectives=["accuracy", "speed"],
    custom_scorer=custom_balanced_accuracy
).quick_classify(X_train, y_train)

# Test the model
y_pred = automl.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Test Accuracy: {test_accuracy:.4f}")

# ============================================================================
# STEP 8: Performance Comparison (No Functions!)
# ============================================================================

print("\n📊 STEP 8: Performance Comparison")
print("=" * 50)

# Generate data
X, y = make_classification(n_samples=250, n_features=6, n_informative=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Dataset shape: {X.shape}")

# Compare different Optuna configurations
configurations = [
    ("TPE Basic", {"sampler": "TPE"}),
    ("TPE + Pruning", {"sampler": "TPE", "pruner": "MedianPruner"}),
    ("NSGAII", {"sampler": "NSGAII"}),
    ("Random", {"sampler": "Random"}),
    ("CmaEs", {"sampler": "CmaEs"})
]

results = {}

for name, config in configurations:
    print(f"\n🔬 Testing {name}:")
    
    try:
        automl = SimpleAutoML(
            learners=["random_forest", "logistic_regression"],
            task="classification",
            n_trials=3,
            cv=3
        )
        
        automl.with_advanced_search(searcher_type="optuna", **config).quick_classify(X_train, y_train)
        
        y_pred = automl.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            "accuracy": accuracy,
            "best_score": automl.best_score,
            "best_learner": automl.best_config.get('learner_name', 'unknown')
        }
        
        print(f"   ✅ Accuracy: {accuracy:.4f}")
        print(f"   📈 Best CV Score: {automl.best_score:.4f}")
        print(f"   🏆 Best Learner: {automl.best_config.get('learner_name', 'unknown')}")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        results[name] = {"error": str(e)}

# Summary
print("\n📋 Optuna Configuration Comparison:")
for name, result in results.items():
    if "error" not in result:
        print(f"   {name}: {result['accuracy']:.4f} accuracy, best: {result['best_learner']}")
    else:
        print(f"   {name}: Failed - {result['error']}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 60)
print("🎉 OPTUNA SEARCH TUTORIAL COMPLETED!")
print("=" * 60)
print("✅ You've explored advanced Optuna features with ZERO function definitions!")
print("✅ Method chaining, custom objectives, and multi-objective optimization!")
print("✅ Pandas-style interface makes advanced optimization simple!")
print("\n💡 Key Takeaways:")
print("   • No function definitions needed")
print("   • Method chaining for complex configurations")
print("   • Custom objectives and multi-objective optimization")
print("   • Auto-execution and smart defaults")
print("   • Advanced optimization techniques made simple")
print("\n🚀 Ready to use advanced Optuna features like pandas!")
