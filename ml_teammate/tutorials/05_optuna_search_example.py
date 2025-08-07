# tutorials/05_optuna_search_example.py
"""
05_optuna_search_example.py
---------------------------
Demonstrate search capabilities in MLTeammate with different searcher types.

This tutorial showcases:
1. Different search algorithms (Random, FLAML if available)
2. Search parameter optimization
3. Performance comparison between searchers
4. Advanced search configurations

Perfect for users who want to explore different optimization strategies!
"""

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# Import the MLTeammate API
from ml_teammate.interface.api import MLTeammate

# ============================================================================
# STEP 1: Explore Available Search Algorithms
# ============================================================================

print("🔍 STEP 1: Available Search Algorithms in MLTeammate")
print("=" * 50)

# Available search algorithms
available_searchers = {
    "random": "Random search - explores parameter space randomly",
    "flaml": "FLAML search - Microsoft's fast AutoML library (if available)",
    "optuna": "Optuna search - advanced optimization (fallback to random if not available)"
}

print("📊 Available Search Algorithms:")
for i, (name, description) in enumerate(available_searchers.items(), 1):
    print(f"   {i:2d}. {name}: {description}")

print(f"\n📋 Total Available Searchers: {len(available_searchers)}")
print("💡 You can specify searcher_type in MLTeammate API!")

# ============================================================================
# STEP 2: Basic Search Algorithm Comparison
# ============================================================================

print("\n🔬 STEP 2: Basic Search Algorithm Comparison")
print("=" * 50)

# Generate sample data
X, y = make_classification(
    n_samples=800,
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

# Test different search algorithms
search_algorithms = ["random", "flaml"]
results = {}

for searcher_type in search_algorithms:
    print(f"\n🧪 Testing {searcher_type} search:")
    
    try:
        automl = MLTeammate(
            learners=["random_forest", "gradient_boosting", "logistic_regression"],
            task="classification",
            searcher_type=searcher_type,
            n_trials=8,
            cv_folds=3,
            random_state=42
        )
        
        automl.fit(X_train, y_train)
        
        # Test the model
        y_pred = automl.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        results[searcher_type] = {
            "best_cv_score": automl.best_score,
            "test_accuracy": test_accuracy,
            "best_config": automl.best_config,
            "best_learner": automl.best_config.get('learner_name', 'unknown')
        }
        
        print(f"   ✅ Best CV Score: {automl.best_score:.4f}")
        print(f"   🎯 Test Accuracy: {test_accuracy:.4f}")
        print(f"   🏆 Best Learner: {automl.best_config.get('learner_name', 'unknown')}")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        results[searcher_type] = {"error": str(e)}

# Compare results
print(f"\n📊 Search Algorithm Comparison:")
print(f"{'Algorithm':<12} {'CV Score':<10} {'Test Score':<10} {'Best Learner':<15}")
print("-" * 55)
for searcher, result in results.items():
    if "error" not in result:
        print(f"{searcher:<12} {result['best_cv_score']:<10.4f} {result['test_accuracy']:<10.4f} {result['best_learner']:<15}")
    else:
        print(f"{searcher:<12} {'Failed':<10} {'N/A':<10} {'N/A':<15}")

# ============================================================================
# STEP 3: Search with Different Trial Counts
# ============================================================================

print("\n🎯 STEP 3: Search with Different Trial Counts")
print("=" * 50)

# Generate data
X, y = make_classification(n_samples=600, n_features=15, n_informative=8, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Dataset shape: {X.shape}")

# Test different trial counts
trial_counts = [3, 5, 10, 15]
trial_results = {}

for n_trials in trial_counts:
    print(f"\n🔬 Testing with {n_trials} trials:")
    
    try:
        automl = MLTeammate(
            learners=["random_forest", "gradient_boosting"],
            task="classification",
            searcher_type="random",
            n_trials=n_trials,
            cv_folds=3,
            random_state=42
        )
        
        automl.fit(X_train, y_train)
        
        # Test the model
        y_pred = automl.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        trial_results[n_trials] = {
            "best_cv_score": automl.best_score,
            "test_accuracy": test_accuracy,
            "best_learner": automl.best_config.get('learner_name', 'unknown')
        }
        
        print(f"   ✅ Best CV Score: {automl.best_score:.4f}")
        print(f"   🎯 Test Accuracy: {test_accuracy:.4f}")
        print(f"   🏆 Best Learner: {automl.best_config.get('learner_name', 'unknown')}")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        trial_results[n_trials] = {"error": str(e)}

# Show trial count impact
print(f"\n📊 Trial Count Impact:")
print(f"{'Trials':<8} {'CV Score':<10} {'Test Score':<10} {'Best Learner':<15}")
print("-" * 50)
for trials, result in trial_results.items():
    if "error" not in result:
        print(f"{trials:<8} {result['best_cv_score']:<10.4f} {result['test_accuracy']:<10.4f} {result['best_learner']:<15}")
    else:
        print(f"{trials:<8} {'Failed':<10} {'N/A':<10} {'N/A':<15}")

# ============================================================================
# STEP 4: Cross-Validation Strategy Comparison
# ============================================================================

print("\n⚡ STEP 4: Cross-Validation Strategy Comparison")
print("=" * 50)

# Generate data
X, y = make_classification(n_samples=500, n_features=12, n_informative=6, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Dataset shape: {X.shape}")

# Test different CV strategies
cv_strategies = [2, 3, 5, None]  # None means no CV (train/test split)
cv_results = {}

for cv_folds in cv_strategies:
    cv_name = f"{cv_folds}-fold CV" if cv_folds else "Train/Test Split"
    print(f"\n🔬 Testing {cv_name}:")
    
    try:
        automl = MLTeammate(
            learners=["random_forest", "logistic_regression"],
            task="classification",
            searcher_type="random",
            n_trials=6,
            cv_folds=cv_folds,
            random_state=42
        )
        
        automl.fit(X_train, y_train)
        
        # Test the model
        y_pred = automl.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        cv_results[cv_name] = {
            "best_score": automl.best_score,
            "test_accuracy": test_accuracy,
            "best_learner": automl.best_config.get('learner_name', 'unknown')
        }
        
        print(f"   ✅ Best Score: {automl.best_score:.4f}")
        print(f"   🎯 Test Accuracy: {test_accuracy:.4f}")
        print(f"   🏆 Best Learner: {automl.best_config.get('learner_name', 'unknown')}")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        cv_results[cv_name] = {"error": str(e)}

# Show CV strategy impact
print(f"\n📊 Cross-Validation Strategy Impact:")
for strategy, result in cv_results.items():
    if "error" not in result:
        print(f"   {strategy}: {result['test_accuracy']:.4f} test accuracy, best: {result['best_learner']}")
    else:
        print(f"   {strategy}: Failed - {result['error']}")

# ============================================================================
# STEP 5: Regression Search Comparison
# ============================================================================

print("\n📈 STEP 5: Regression Search Comparison")
print("=" * 50)

# Generate regression data
X_reg, y_reg = make_regression(n_samples=500, n_features=15, n_informative=8, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

print(f"📊 Dataset shape: {X_reg.shape}")

# Test regression with different search algorithms
reg_results = {}

for searcher_type in ["random", "flaml"]:
    print(f"\n🔬 Testing {searcher_type} search for regression:")
    
    try:
        automl_reg = MLTeammate(
            learners=["random_forest_regressor", "linear_regression", "ridge"],
            task="regression",
            searcher_type=searcher_type,
            n_trials=6,
            cv_folds=3,
            random_state=42
        )
        
        automl_reg.fit(X_train_reg, y_train_reg)
        
        # Test the model
        y_pred_reg = automl_reg.predict(X_test_reg)
        test_r2 = r2_score(y_test_reg, y_pred_reg)
        
        reg_results[searcher_type] = {
            "best_score": automl_reg.best_score,
            "test_r2": test_r2,
            "best_learner": automl_reg.best_config.get('learner_name', 'unknown')
        }
        
        print(f"   ✅ Best CV Score: {automl_reg.best_score:.4f}")
        print(f"   🎯 Test R² Score: {test_r2:.4f}")
        print(f"   🏆 Best Learner: {automl_reg.best_config.get('learner_name', 'unknown')}")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        reg_results[searcher_type] = {"error": str(e)}

# Compare regression results
print(f"\n📊 Regression Search Comparison:")
for searcher, result in reg_results.items():
    if "error" not in result:
        print(f"   {searcher}: {result['test_r2']:.4f} R² score, best: {result['best_learner']}")
    else:
        print(f"   {searcher}: Failed - {result['error']}")

# ============================================================================
# STEP 6: MLflow Integration with Search
# ============================================================================

print("\n🚀 STEP 6: MLflow Integration with Search")
print("=" * 50)

# Generate data
X, y = make_classification(n_samples=400, n_features=10, n_informative=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Dataset shape: {X.shape}")

# Test search with MLflow tracking
print("\n🔬 Search with MLflow tracking:")

try:
    automl = MLTeammate(
        learners=["random_forest", "gradient_boosting"],
        task="classification",
        searcher_type="random",
        n_trials=6,
        cv_folds=3,
        enable_mlflow=True,  # Enable MLflow tracking
        random_state=42
    )
    
    automl.fit(X_train, y_train)
    
    # Test the model
    y_pred = automl.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"   ✅ Best CV Score: {automl.best_score:.4f}")
    print(f"   🎯 Test Accuracy: {test_accuracy:.4f}")
    print(f"   🏆 Best Learner: {automl.best_config.get('learner_name', 'unknown')}")
    print(f"   📊 MLflow tracking enabled - all trials logged")
    
    # Show summary
    print(f"\n📋 MLflow Experiment Summary:")
    summary = automl.summary()
    for key, value in summary.items():
        print(f"      {key}: {value}")
        
except Exception as e:
    print(f"   ❌ MLflow integration failed: {e}")

# ============================================================================
# STEP 7: Comprehensive Search Comparison
# ============================================================================

print("\n📊 STEP 7: Comprehensive Search Comparison")
print("=" * 50)

# Generate final test data
X, y = make_classification(n_samples=300, n_features=8, n_informative=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Dataset shape: {X.shape}")

# Test comprehensive configurations
configurations = [
    ("Random Search (Basic)", {"searcher_type": "random", "n_trials": 5}),
    ("Random Search (Extended)", {"searcher_type": "random", "n_trials": 10}),
    ("FLAML Search", {"searcher_type": "flaml", "n_trials": 5}),
    ("Random + MLflow", {"searcher_type": "random", "n_trials": 5, "enable_mlflow": True}),
]

final_results = {}

for name, config in configurations:
    print(f"\n🔬 Testing {name}:")
    
    try:
        automl = MLTeammate(
            learners=["random_forest", "logistic_regression", "gradient_boosting"],
            task="classification",
            cv_folds=3,
            random_state=42,
            **config
        )
        
        automl.fit(X_train, y_train)
        
        y_pred = automl.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        final_results[name] = {
            "accuracy": accuracy,
            "best_score": automl.best_score,
            "best_learner": automl.best_config.get('learner_name', 'unknown')
        }
        
        print(f"   ✅ Accuracy: {accuracy:.4f}")
        print(f"   📈 Best CV Score: {automl.best_score:.4f}")
        print(f"   🏆 Best Learner: {automl.best_config.get('learner_name', 'unknown')}")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        final_results[name] = {"error": str(e)}

# Final comparison
print("\n📋 Final Search Configuration Comparison:")
for name, result in final_results.items():
    if "error" not in result:
        print(f"   {name}: {result['accuracy']:.4f} accuracy, CV: {result['best_score']:.4f}, best: {result['best_learner']}")
    else:
        print(f"   {name}: Failed - {result['error']}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 60)
print("🎉 SEARCH OPTIMIZATION TUTORIAL COMPLETED!")
print("=" * 60)
print("✅ You've explored different search algorithms!")
print("✅ Compared random search vs FLAML search!")
print("✅ Tested different trial counts and CV strategies!")
print("✅ Integrated MLflow experiment tracking!")
print("\n💡 Key Takeaways:")
print("   • Different search algorithms have different strengths")
print("   • More trials generally lead to better results")
print("   • Cross-validation provides more robust estimates")
print("   • MLflow integration enables experiment tracking")
print("   • Random search is simple and effective")
print("   • FLAML search can be more efficient for larger spaces")
print("\n🚀 Ready to optimize your AutoML searches!")

if __name__ == "__main__":
    pass
