# tutorials/06_simple_api_example.py
"""
06_simple_api_example.py
------------------------
Demonstrate the simplified MLTeammate API that requires NO custom code.

This tutorial shows how users can run AutoML experiments by simply specifying
learner names as strings, without writing any custom classes, functions, or
configuration spaces.

Perfect for Jupyter notebook users and beginners!
"""

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# Import the simplified API
from ml_teammate.interface.simple_api import (
    SimpleAutoML,
    quick_classification,
    quick_regression,
    list_available_learners,
    get_learner_info
)

# ============================================================================
# STEP 1: Explore Available Learners
# ============================================================================

def explore_learners():
    """Show what learners are available."""
    print("🔍 Available Learners in MLTeammate")
    print("=" * 50)
    
    learners = list_available_learners()
    
    print("📊 Classification Learners:")
    for i, learner in enumerate(learners["classification"], 1):
        print(f"   {i:2d}. {learner}")
    
    print("\n📈 Regression Learners:")
    for i, learner in enumerate(learners["regression"], 1):
        print(f"   {i:2d}. {learner}")
    
    print(f"\n📋 Total Available Learners: {len(learners['all'])}")
    print("💡 You can use any of these by simply specifying their names as strings!")


# ============================================================================
# STEP 2: Simple Classification Example
# ============================================================================

def simple_classification_example():
    """Run a simple classification experiment with minimal code."""
    
    print("\n🚀 Simple Classification Example")
    print("=" * 50)
    
    # Generate sample data
    print("📊 Creating sample classification dataset...")
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
    
    print(f"📈 Dataset shape: {X.shape}")
    print(f"🎯 Classes: {np.unique(y)}")
    
    # Run AutoML with just learner names!
    print("\n🔬 Running AutoML with 3 popular learners...")
    
    automl = SimpleAutoML(
        learners=["random_forest", "logistic_regression", "xgboost"],
        task="classification",
        n_trials=10,
        cv=3
    )
    
    # Fit the model (this is all you need!)
    automl.fit(X_train, y_train)
    
    # Make predictions
    y_pred = automl.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    # Show results
    print(f"\n🎉 Results:")
    print(f"📈 Best CV Score: {automl.best_score:.4f}")
    print(f"🎯 Test Accuracy: {test_accuracy:.4f}")
    print(f"🏆 Best Learner: {automl.best_config.get('learner_name', 'unknown')}")
    print(f"⚙️  Best Config: {automl.best_config}")
    
    return automl, test_accuracy


# ============================================================================
# STEP 3: Quick Classification Function
# ============================================================================

def quick_classification_example():
    """Demonstrate the ultra-simple quick_classification function."""
    
    print("\n⚡ Quick Classification Example")
    print("=" * 50)
    
    # Generate data
    X, y = make_classification(n_samples=500, n_features=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("🚀 Running quick classification with just one line!")
    
    # This is all you need - one function call!
    automl = quick_classification(
        X_train, y_train,
        learners=["random_forest", "svm"],
        n_trials=5,
        cv=3
    )
    
    # Evaluate
    y_pred = automl.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"🎯 Test Accuracy: {accuracy:.4f}")
    print(f"📊 Best Score: {automl.best_score:.4f}")
    
    return automl, accuracy


# ============================================================================
# STEP 4: Regression Example
# ============================================================================

def regression_example():
    """Demonstrate regression with the simple API."""
    
    print("\n📈 Regression Example")
    print("=" * 50)
    
    # Generate regression data
    X, y = make_regression(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        noise=0.1,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"📊 Dataset shape: {X.shape}")
    print(f"📈 Target range: {y.min():.2f} to {y.max():.2f}")
    
    # Run regression AutoML
    automl = SimpleAutoML(
        learners=["random_forest_regressor", "linear_regression", "ridge"],
        task="regression",
        n_trials=10,
        cv=3
    )
    
    automl.fit(X_train, y_train)
    
    # Evaluate
    y_pred = automl.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n🎉 Regression Results:")
    print(f"📈 Best CV Score: {automl.best_score:.4f}")
    print(f"🎯 Test MSE: {mse:.4f}")
    print(f"📊 Test R²: {r2:.4f}")
    print(f"🏆 Best Learner: {automl.best_config.get('learner_name', 'unknown')}")
    
    return automl, mse, r2


# ============================================================================
# STEP 5: Single Learner Example
# ============================================================================

def single_learner_example():
    """Show how to use just one learner."""
    
    print("\n🎯 Single Learner Example")
    print("=" * 50)
    
    # Generate data
    X, y = make_classification(n_samples=300, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("🔬 Running AutoML with just Random Forest...")
    
    # Just specify the learner name as a string!
    automl = SimpleAutoML(
        learners="random_forest",  # Single learner as string
        task="classification",
        n_trials=5,
        cv=3
    )
    
    automl.fit(X_train, y_train)
    
    y_pred = automl.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"🎯 Test Accuracy: {accuracy:.4f}")
    print(f"📊 Best Score: {automl.best_score:.4f}")
    
    return automl, accuracy


# ============================================================================
# STEP 6: MLflow Integration Example
# ============================================================================

def mlflow_example():
    """Demonstrate MLflow integration with the simple API."""
    
    print("\n📊 MLflow Integration Example")
    print("=" * 50)
    
    # Generate data
    X, y = make_classification(n_samples=500, n_features=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("🔬 Running AutoML with MLflow tracking...")
    
    automl = SimpleAutoML(
        learners=["random_forest", "logistic_regression", "xgboost"],
        task="classification",
        n_trials=5,
        cv=3,
        use_mlflow=True,
        experiment_name="simple_api_demo",
        save_artifacts=True
    )
    
    automl.fit(X_train, y_train)
    
    y_pred = automl.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"🎯 Test Accuracy: {accuracy:.4f}")
    print(f"📊 Best Score: {automl.best_score:.4f}")
    print(f"📁 MLflow experiment: simple_api_demo")
    print(f"💾 Artifacts saved to: {automl.output_dir}")
    
    return automl, accuracy


# ============================================================================
# STEP 7: Learner Information Example
# ============================================================================

def learner_info_example():
    """Show how to get information about learners."""
    
    print("\nℹ️ Learner Information Example")
    print("=" * 50)
    
    # Get info about different learners
    learners_to_check = ["random_forest", "logistic_regression", "svm", "xgboost"]
    
    for learner in learners_to_check:
        info = get_learner_info(learner)
        
        if "error" not in info:
            print(f"\n📋 {learner.upper()}:")
            print(f"   Parameters: {len(info['parameters'])}")
            print(f"   Classification: {info['is_classification']}")
            print(f"   Regression: {info['is_regression']}")
            print(f"   Sample params: {list(info['parameters'])[:3]}...")
        else:
            print(f"\n❌ {learner}: {info['error']}")


# ============================================================================
# STEP 8: Results Summary Example
# ============================================================================

def results_summary_example():
    """Show how to get a comprehensive results summary."""
    
    print("\n📊 Results Summary Example")
    print("=" * 50)
    
    # Run a quick experiment
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    automl = SimpleAutoML(
        learners=["random_forest", "logistic_regression"],
        task="classification",
        n_trials=3,
        cv=2
    )
    
    automl.fit(X_train, y_train)
    
    # Get comprehensive results summary
    summary = automl.get_results_summary()
    
    print("📋 Experiment Summary:")
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    return automl, summary


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("🚀 MLTeammate Simple API Tutorial")
    print("=" * 60)
    print("This tutorial demonstrates how to use MLTeammate WITHOUT writing")
    print("any custom classes, functions, or configuration spaces!")
    print("=" * 60)
    
    # Step 1: Explore available learners
    explore_learners()
    
    # Step 2: Simple classification
    automl1, acc1 = simple_classification_example()
    
    # Step 3: Quick classification
    automl2, acc2 = quick_classification_example()
    
    # Step 4: Regression
    automl3, mse, r2 = regression_example()
    
    # Step 5: Single learner
    automl4, acc4 = single_learner_example()
    
    # Step 6: MLflow integration
    try:
        automl5, acc5 = mlflow_example()
    except Exception as e:
        print(f"⚠️ MLflow example skipped: {e}")
    
    # Step 7: Learner information
    learner_info_example()
    
    # Step 8: Results summary
    automl6, summary = results_summary_example()
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 TUTORIAL COMPLETED!")
    print("=" * 60)
    print("✅ You've successfully used MLTeammate's Simple API!")
    print("✅ No custom classes or functions were written!")
    print("✅ All complexity was handled automatically!")
    print("\n💡 Key Takeaways:")
    print("   • Just specify learner names as strings")
    print("   • Framework handles all the complexity")
    print("   • Perfect for Jupyter notebooks and beginners")
    print("   • Full MLflow integration available")
    print("   • Comprehensive results and artifacts")
    print("\n🚀 Ready to use MLTeammate in your own projects!") 