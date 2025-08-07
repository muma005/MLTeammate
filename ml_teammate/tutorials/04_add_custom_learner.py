# tutorials/04_add_custom_learner.py
"""
04_add_custom_learner.py
------------------------
Demonstrate how to use MLTeammate with different learner combinations.

This tutorial shows:
1. How to use available learners with different configurations
2. How to compare different learner combinations
3. How to use the MLTeammate API for custom experiments
4. How to get comprehensive results

Perfect for users who want to explore different learner options!
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import the MLTeammate API
from ml_teammate.interface.api import MLTeammate

# ============================================================================
# STEP 1: Explore Available Learners
# ============================================================================

print("🔍 STEP 1: Available Learners in MLTeammate")
print("=" * 50)

# Available learners for classification
available_learners = {
    "classification": ["random_forest", "logistic_regression", "svm", "gradient_boosting"],
    "regression": ["linear_regression", "ridge", "random_forest_regressor", "gradient_boosting_regressor"]
}

print("📊 Classification Learners:")
for i, learner in enumerate(available_learners["classification"], 1):
    print(f"   {i:2d}. {learner}")

print("\n📈 Regression Learners:")
for i, learner in enumerate(available_learners["regression"], 1):
    print(f"   {i:2d}. {learner}")

print(f"\n📋 Total Available Learners: {len(available_learners['classification']) + len(available_learners['regression'])}")
print("💡 You can use any of these by simply specifying their names as strings!")

# ============================================================================
# STEP 2: Basic Learner Comparison
# ============================================================================

print("\n🎯 STEP 2: Basic Learner Comparison")
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

print(f"📊 Dataset shape: {X.shape}")
print(f"🎯 Classes: {np.unique(y)}")

# Method 1: Use tree-based learners
print("\n🔬 Method 1: Tree-based learners")
automl = MLTeammate(
    learners=["random_forest", "gradient_boosting"],  # Available tree-based learners
    task="classification",
    n_trials=8,
    cv_folds=3,
    random_state=42
)
automl.fit(X_train, y_train)

# Test the model
y_pred = automl.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"📈 Best CV Score: {automl.best_score:.4f}")
print(f"🎯 Test Accuracy: {test_accuracy:.4f}")
print(f"🏆 Best Learner: {automl.best_config.get('learner_name', 'unknown')}")

# Method 2: Use linear learners for comparison
print("\n🔬 Method 2: Linear learners")
automl2 = MLTeammate(
    learners=["logistic_regression", "svm"], 
    task="classification",
    n_trials=5,
    cv_folds=3,
    random_state=42
)
automl2.fit(X_train, y_train)

# Test the model
y_pred2 = automl2.predict(X_test)
test_accuracy2 = accuracy_score(y_test, y_pred2)
print(f"📈 Best CV Score: {automl2.best_score:.4f}")
print(f"🎯 Test Accuracy: {test_accuracy2:.4f}")
print(f"🏆 Best Learner: {automl2.best_config.get('learner_name', 'unknown')}")

# ============================================================================
# STEP 3: Compare All Available Learners
# ============================================================================

print("\n⚡ STEP 3: Compare All Available Learners")
print("=" * 50)

# Generate data
X, y = make_classification(n_samples=800, n_features=15, n_informative=8, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Dataset shape: {X.shape}")

# Use all available learners
print("\n🔬 Using all available classification learners:")
automl = MLTeammate(
    learners=["random_forest", "gradient_boosting", "logistic_regression", "svm"], 
    task="classification",
    n_trials=8,
    cv_folds=3,
    random_state=42
)
automl.fit(X_train, y_train)

# Test the model
y_pred = automl.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"📈 Best CV Score: {automl.best_score:.4f}")
print(f"🎯 Test Accuracy: {test_accuracy:.4f}")
print(f"🏆 Best Learner: {automl.best_config.get('learner_name', 'unknown')}")

# Show summary
print("\n📋 Results Summary:")
summary = automl.summary()
for key, value in summary.items():
    print(f"   {key}: {value}")

# ============================================================================
# STEP 4: MLflow Integration with Learner Comparison
# ============================================================================

print("\n🔧 STEP 4: MLflow Integration with Learner Comparison")
print("=" * 50)

# Generate data
X, y = make_classification(n_samples=600, n_features=12, n_informative=6, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Dataset shape: {X.shape}")

# Compare with MLflow tracking enabled
print("\n🔬 Experiment with MLflow tracking:")
automl = MLTeammate(
    learners=["random_forest", "gradient_boosting"],
    task="classification",
    n_trials=6,
    cv_folds=3,
    enable_mlflow=True,  # Enable MLflow tracking
    random_state=42
)
automl.fit(X_train, y_train)

# Test the model
y_pred = automl.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"📈 Best CV Score: {automl.best_score:.4f}")
print(f"🎯 Test Accuracy: {test_accuracy:.4f}")
print(f"🏆 Best Learner: {automl.best_config.get('learner_name', 'unknown')}")

# ============================================================================
# STEP 5: Learner Performance Comparison
# ============================================================================

print("\n🔗 STEP 5: Learner Performance Comparison")
print("=" * 50)

# Generate data
X, y = make_classification(n_samples=500, n_features=10, n_informative=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Dataset shape: {X.shape}")

# Compare different learner combinations
learner_combinations = [
    ("Tree-based models", ["random_forest", "gradient_boosting"]),
    ("Linear models", ["logistic_regression", "svm"]),
    ("Fast models", ["logistic_regression", "random_forest"]),
    ("Robust models", ["gradient_boosting", "svm"])
]

results = {}

for name, learners in learner_combinations:
    print(f"\n🔬 Testing {name}:")
    
    try:
        automl = MLTeammate(
            learners=learners, 
            task="classification",
            n_trials=4,
            cv_folds=3,
            random_state=42
        )
        automl.fit(X_train, y_train)
        
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
print("\n📋 Comparison Summary:")
for name, result in results.items():
    if "error" not in result:
        print(f"   {name}: {result['accuracy']:.4f} accuracy, best: {result['best_learner']}")
    else:
        print(f"   {name}: Failed - {result['error']}")

# ============================================================================
# STEP 6: Cross-Validation Comparison
# ============================================================================

print("\n🧠 STEP 6: Cross-Validation Comparison")
print("=" * 50)

# Generate data
X, y = make_classification(n_samples=400, n_features=8, n_informative=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Dataset shape: {X.shape}")

# Compare different CV strategies
cv_strategies = [2, 3, 5]

for cv_folds in cv_strategies:
    print(f"\n🔬 Testing {cv_folds}-fold cross-validation:")
    
    automl = MLTeammate(
        learners=["random_forest", "logistic_regression"],
        task="classification",
        n_trials=4,
        cv_folds=cv_folds,
        random_state=42
    )
    automl.fit(X_train, y_train)
    
    y_pred = automl.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"   📈 Best CV Score: {automl.best_score:.4f}")
    print(f"   🎯 Test Accuracy: {test_accuracy:.4f}")
    print(f"   🏆 Best Learner: {automl.best_config.get('learner_name', 'unknown')}")

# ============================================================================
# STEP 7: Final Results Summary
# ============================================================================

print("\n📋 STEP 7: Final Results Summary")
print("=" * 50)

# Generate data for final test
X, y = make_classification(n_samples=300, n_features=6, n_informative=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Dataset shape: {X.shape}")

# Final comprehensive experiment
automl = MLTeammate(
    learners=["random_forest", "gradient_boosting", "logistic_regression"], 
    task="classification",
    n_trials=6,
    cv_folds=3,
    random_state=42
)
automl.fit(X_train, y_train)

# Get comprehensive results summary
print("\n📋 Final Results Summary:")
summary = automl.summary()
for key, value in summary.items():
    print(f"   {key}: {value}")

# Final test
y_pred = automl.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Final Test Accuracy: {final_accuracy:.4f}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 60)
print("🎉 LEARNER COMPARISON TUTORIAL COMPLETED!")
print("=" * 60)
print("✅ You've explored different learner combinations!")
print("✅ Compared tree-based vs linear models!")
print("✅ Used cross-validation strategies!")
print("✅ Integrated MLflow experiment tracking!")
print("\n💡 Key Takeaways:")
print("   • Different learners excel on different datasets")
print("   • Tree-based models (Random Forest, Gradient Boosting) often perform well")
print("   • Linear models (Logistic Regression, SVM) are fast and interpretable")
print("   • Cross-validation provides more robust performance estimates")
print("   • MLflow integration enables experiment tracking")
print("\n🚀 Ready to choose the best learners for your data!")

if __name__ == "__main__":
    pass
