# tutorials/07_advanced_search_example.py
"""
07_advanced_search_example.py
------------------------------
Demonstrate advanced search capabilities in MLTeammate with pandas-style interface.

This tutorial showcases:
1. FLAML-based hyperparameter optimization
2. Early Convergence Indicators (ECI)
3. Time-bounded optimization
4. Resource-aware optimization
5. Multi-objective optimization

Perfect for users who want to explore advanced optimization techniques!
"""

import numpy as np
import time
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# Import MLTeammate components
from ml_teammate.interface import SimpleAutoML
from ml_teammate.search import (
    FLAMLSearcher,
    FLAMLTimeBudgetSearcher,
    FLAMLResourceAwareSearcher,
    EarlyConvergenceIndicator,
    AdaptiveECI,
    MultiObjectiveECI,
    get_searcher,
    get_eci,
    list_available_searchers,
    list_available_eci_types
)

# ============================================================================
# STEP 1: Explore Available Search Components (No Functions!)
# ============================================================================

print("🔍 STEP 1: Explore Available Search Components")
print("=" * 60)

# List searchers
searchers = list_available_searchers()
print("📊 Searchers:")
for name, info in searchers.items():
    print(f"   • {name}: {info['description']}")
    print(f"     Features: {', '.join(info['features'])}")
    print(f"     Dependencies: {', '.join(info['dependencies'])}")
    print()

# List ECI types
eci_types = list_available_eci_types()
print("🎯 Early Convergence Indicators:")
for name, info in eci_types.items():
    print(f"   • {name}: {info['description']}")
    print(f"     Methods: {', '.join(info['methods'])}")
    print(f"     Features: {', '.join(info['features'])}")
    print()

# ============================================================================
# STEP 2: FLAML Searcher Example (No Functions!)
# ============================================================================

print("\n🚀 STEP 2: FLAML Searcher Example")
print("=" * 50)

# Generate sample data
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

print("📊 Dataset shape:", X.shape)
print("🎯 Classes:", np.unique(y))

# Method 1: Using SimpleAutoML with FLAML
print("\n🔬 Method 1: SimpleAutoML with FLAML")
automl = SimpleAutoML()
automl.with_flaml(time_budget=30).quick_classify(X_train, y_train)  # Auto-executes!

# Test the model
y_pred = automl.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Test Accuracy: {test_accuracy:.4f}")

# Method 2: Direct FLAML searcher
print("\n🔬 Method 2: Direct FLAML searcher")
config_space = {
    "random_forest": {
        "n_estimators": {"type": "int", "bounds": [50, 200]},
        "max_depth": {"type": "int", "bounds": [3, 10]},
        "min_samples_split": {"type": "int", "bounds": [2, 10]}
    },
    "logistic_regression": {
        "C": {"type": "float", "bounds": [0.1, 10.0]},
        "max_iter": {"type": "int", "bounds": [100, 1000]}
    }
}

flaml_searcher = get_searcher("flaml", config_spaces=config_space, time_budget=20)
flaml_searcher.fit(X_train, y_train, task="classification")

best_result = flaml_searcher.get_best()
print(f"🏆 Best FLAML Result: {best_result}")

# ============================================================================
# STEP 3: Early Convergence Indicator (No Functions!)
# ============================================================================

print("\n🎯 STEP 3: Early Convergence Indicator")
print("=" * 50)

# Generate data
X, y = make_classification(n_samples=800, n_features=15, n_informative=8, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("📊 Dataset shape:", X.shape)

# Method 1: Using ECI with SimpleAutoML
print("\n🔬 Method 1: ECI with SimpleAutoML")
automl = SimpleAutoML()
automl.with_eci(eci_type="standard", window_size=5, patience=3).quick_classify(X_train, y_train)

# Method 2: Direct ECI usage
print("\n🔬 Method 2: Direct ECI usage")
eci = get_eci("standard", window_size=5, patience=3)

# Simulate some trials
scores = [0.7, 0.72, 0.71, 0.73, 0.72, 0.73, 0.73, 0.73, 0.73, 0.73]
for i, score in enumerate(scores):
    eci.add_trial(f"trial_{i}", score)
    if eci.should_stop():
        print(f"🛑 ECI detected convergence at trial {i+1}")
        break

convergence_info = eci.get_convergence_info()
print(f"📊 Convergence Info: {convergence_info}")

# ============================================================================
# STEP 4: Multi-Objective ECI (No Functions!)
# ============================================================================

print("\n🎯 STEP 4: Multi-Objective ECI")
print("=" * 50)

# Generate data
X, y = make_classification(n_samples=600, n_features=12, n_informative=6, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("📊 Dataset shape:", X.shape)

# Method 1: Multi-objective ECI with SimpleAutoML
print("\n🔬 Method 1: Multi-objective ECI with SimpleAutoML")
automl = SimpleAutoML()
automl.with_eci(eci_type="multi_objective", objectives=["accuracy", "speed"]).quick_classify(X_train, y_train)

# Method 2: Direct multi-objective ECI
print("\n🔬 Method 2: Direct multi-objective ECI")
multi_eci = get_eci("multi_objective", objectives=["accuracy", "speed"])

# Simulate multi-objective trials
trials = [
    {"accuracy": 0.75, "speed": 0.8},
    {"accuracy": 0.78, "speed": 0.7},
    {"accuracy": 0.76, "speed": 0.9},
    {"accuracy": 0.77, "speed": 0.75},
    {"accuracy": 0.77, "speed": 0.75}
]

for i, metrics in enumerate(trials):
    multi_eci.add_trial(f"trial_{i}", metrics)
    if multi_eci.should_stop():
        print(f"🛑 Multi-objective ECI detected convergence at trial {i+1}")
        break

# ============================================================================
# STEP 5: Resource-Aware FLAML (No Functions!)
# ============================================================================

print("\n⚡ STEP 5: Resource-Aware FLAML")
print("=" * 50)

# Generate data
X, y = make_classification(n_samples=500, n_features=10, n_informative=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("📊 Dataset shape:", X.shape)

# Method 1: Resource-aware FLAML with SimpleAutoML
print("\n🔬 Method 1: Resource-aware FLAML with SimpleAutoML")
automl = SimpleAutoML()
automl.with_flaml(time_budget=15, max_iter=50).quick_classify(X_train, y_train)

# Method 2: Direct resource-aware FLAML
print("\n🔬 Method 2: Direct resource-aware FLAML")
resource_searcher = get_searcher(
    "flaml_resource_aware", 
    config_spaces=config_space, 
    time_budget=15,
    max_iter=50
)
resource_searcher.fit(X_train, y_train, task="classification")

# ============================================================================
# STEP 6: Factory Functions (No Functions!)
# ============================================================================

print("\n🏭 STEP 6: Factory Functions")
print("=" * 50)

# Generate data
X, y = make_classification(n_samples=400, n_features=8, n_informative=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("📊 Dataset shape:", X.shape)

# Method 1: Using factory functions with SimpleAutoML
print("\n🔬 Method 1: Factory functions with SimpleAutoML")
automl = SimpleAutoML()
automl.with_advanced_search(searcher_type="flaml", time_budget=10).quick_classify(X_train, y_train)

# Method 2: Direct factory function usage
print("\n🔬 Method 2: Direct factory function usage")

# Create different searchers using factory
searchers_to_test = ["optuna", "flaml", "flaml_time_budget"]

for searcher_type in searchers_to_test:
    print(f"\n🔬 Testing {searcher_type} searcher:")
    try:
        searcher = get_searcher(searcher_type, config_spaces=config_space, time_budget=5)
        searcher.fit(X_train, y_train, task="classification")
        best = searcher.get_best()
        print(f"   ✅ {searcher_type} completed successfully")
        print(f"   🏆 Best result: {best.get('score', 'N/A')}")
    except Exception as e:
        print(f"   ❌ {searcher_type} failed: {e}")

# ============================================================================
# STEP 7: Method Chaining (No Functions!)
# ============================================================================

print("\n🔗 STEP 7: Method Chaining")
print("=" * 50)

# Generate data
X, y = make_classification(n_samples=300, n_features=6, n_informative=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("📊 Dataset shape:", X.shape)

# Chain multiple methods together - just like pandas!
print("\n🔬 Method Chaining Examples:")

# Example 1: FLAML + MLflow + ECI
print("\n🔬 Example 1: FLAML + MLflow + ECI")
automl = SimpleAutoML()
automl.with_flaml(time_budget=10).with_mlflow(experiment_name="chaining_demo").with_eci().quick_classify(X_train, y_train)

# Example 2: Resource-aware + ECI
print("\n🔬 Example 2: Resource-aware + ECI")
automl = SimpleAutoML()
automl.with_advanced_search(searcher_type="flaml_resource_aware", time_budget=8).with_eci(eci_type="adaptive").quick_classify(X_train, y_train)

# Example 3: Multi-objective + MLflow
print("\n🔬 Example 3: Multi-objective + MLflow")
automl = SimpleAutoML()
automl.with_eci(eci_type="multi_objective").with_mlflow().quick_classify(X_train, y_train)

# ============================================================================
# STEP 8: Performance Comparison (No Functions!)
# ============================================================================

print("\n📊 STEP 8: Performance Comparison")
print("=" * 50)

# Generate data
X, y = make_classification(n_samples=200, n_features=5, n_informative=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("📊 Dataset shape:", X.shape)

# Compare different searchers
searchers_to_compare = [
    ("Optuna", "optuna"),
    ("FLAML", "flaml"),
    ("FLAML Time Budget", "flaml_time_budget")
]

results = {}

for name, searcher_type in searchers_to_compare:
    print(f"\n🔬 Testing {name}:")
    start_time = time.time()
    
    try:
        automl = SimpleAutoML()
        automl.with_advanced_search(searcher_type=searcher_type, time_budget=5).quick_classify(X_train, y_train)
        
        end_time = time.time()
        duration = end_time - start_time
        
        y_pred = automl.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            "accuracy": accuracy,
            "duration": duration,
            "best_score": automl.best_score
        }
        
        print(f"   ✅ Accuracy: {accuracy:.4f}")
        print(f"   ⏱️ Duration: {duration:.2f}s")
        print(f"   📈 Best CV Score: {automl.best_score:.4f}")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        results[name] = {"error": str(e)}

# Summary
print("\n📋 Performance Summary:")
for name, result in results.items():
    if "error" not in result:
        print(f"   {name}: {result['accuracy']:.4f} accuracy, {result['duration']:.2f}s")
    else:
        print(f"   {name}: Failed - {result['error']}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 60)
print("🎉 ADVANCED SEARCH TUTORIAL COMPLETED!")
print("=" * 60)
print("✅ You've explored advanced search capabilities with ZERO function definitions!")
print("✅ Method chaining, factory functions, and auto-execution!")
print("✅ FLAML, ECI, and multi-objective optimization!")
print("\n💡 Key Takeaways:")
print("   • No function definitions needed")
print("   • Method chaining for complex configurations")
print("   • Factory functions for easy component creation")
print("   • Auto-execution and smart defaults")
print("   • Advanced optimization techniques made simple")
print("\n🚀 Ready to use advanced search features like pandas!") 