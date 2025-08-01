# tutorials/07_advanced_search_example.py
"""
07_advanced_search_example.py
------------------------------
Demonstrate advanced search capabilities in MLTeammate.

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
# STEP 1: Explore Available Search Components
# ============================================================================

def explore_search_components():
    """Show what search components are available."""
    print("🔍 Available Search Components in MLTeammate")
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
# STEP 2: FLAML Searcher Example
# ============================================================================

def flaml_searcher_example():
    """Demonstrate FLAML-based hyperparameter optimization."""
    
    print("\n🚀 FLAML Searcher Example")
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
    
    # Create FLAML searcher
    config_space = {
        "random_forest": {
            "n_estimators": {"type": "int", "bounds": [50, 200]},
            "max_depth": {"type": "int", "bounds": [3, 10]},
            "min_samples_split": {"type": "int", "bounds": [2, 10]}
        },
        "logistic_regression": {
            "C": {"type": "float", "bounds": [0.1, 10.0]},
            "max_iter": {"type": "int", "bounds": [100, 500]}
        }
    }
    
    try:
        # Create FLAML searcher with time budget
        flaml_searcher = FLAMLTimeBudgetSearcher(
            config_spaces=config_space,
            time_budget=30,  # 30 seconds
            metric="accuracy",
            mode="max"
        )
        
        print("🔬 Running FLAML optimization with 30-second time budget...")
        start_time = time.time()
        
        # Run FLAML optimization
        best_config = flaml_searcher.fit(X_train, y_train, task="classification")
        
        optimization_time = time.time() - start_time
        
        print(f"⏱️ Optimization completed in {optimization_time:.2f} seconds")
        print(f"🏆 Best configuration: {best_config}")
        
        # Get optimization summary
        summary = flaml_searcher.get_optimization_summary()
        print(f"📊 Optimization summary:")
        for key, value in summary.items():
            print(f"   {key}: {value}")
        
        return flaml_searcher, best_config
        
    except ImportError as e:
        print(f"⚠️ FLAML not available: {e}")
        print("   Install with: pip install flaml")
        return None, None


# ============================================================================
# STEP 3: Early Convergence Indicator Example
# ============================================================================

def eci_example():
    """Demonstrate Early Convergence Indicators."""
    
    print("\n🎯 Early Convergence Indicator Example")
    print("=" * 50)
    
    # Create different types of ECIs
    standard_eci = EarlyConvergenceIndicator(
        window_size=5,
        min_trials=3,
        improvement_threshold=0.001,
        patience=3,
        convergence_method="moving_average"
    )
    
    adaptive_eci = AdaptiveECI(
        window_size=5,
        min_trials=3,
        improvement_threshold=0.001,
        patience=3
    )
    
    # Simulate optimization progress
    print("🔬 Simulating optimization progress...")
    
    # Simulate scores that improve then plateau
    simulated_scores = [0.5, 0.6, 0.65, 0.67, 0.68, 0.68, 0.68, 0.68, 0.68, 0.68]
    
    for i, score in enumerate(simulated_scores):
        trial_id = f"trial_{i+1}"
        
        # Add to both ECIs
        standard_eci.add_trial(trial_id, score)
        adaptive_eci.add_trial(trial_id, score)
        
        print(f"   Trial {i+1}: Score = {score:.3f}")
        
        # Check convergence
        if standard_eci.should_stop():
            print(f"   🛑 Standard ECI detected convergence after {i+1} trials")
            break
        
        if adaptive_eci.should_stop():
            print(f"   🛑 Adaptive ECI detected convergence after {i+1} trials")
            break
    
    # Get convergence information
    print("\n📊 Convergence Analysis:")
    
    standard_info = standard_eci.get_convergence_info()
    print("   Standard ECI:")
    for key, value in standard_info.items():
        print(f"     {key}: {value}")
    
    adaptive_info = adaptive_eci.get_convergence_info()
    print("   Adaptive ECI:")
    for key, value in adaptive_info.items():
        print(f"     {key}: {value}")
    
    return standard_eci, adaptive_eci


# ============================================================================
# STEP 4: Multi-Objective ECI Example
# ============================================================================

def multi_objective_eci_example():
    """Demonstrate Multi-Objective Early Convergence Indicator."""
    
    print("\n🎯 Multi-Objective ECI Example")
    print("=" * 50)
    
    # Create multi-objective ECI
    multi_eci = MultiObjectiveECI(
        objectives=["accuracy", "speed"],
        window_size=5,
        min_trials=3,
        improvement_threshold=0.001
    )
    
    # Simulate multi-objective optimization
    print("🔬 Simulating multi-objective optimization...")
    
    # Simulate scores for multiple objectives
    simulated_scores = [
        {"accuracy": 0.5, "speed": 0.9},
        {"accuracy": 0.6, "speed": 0.8},
        {"accuracy": 0.65, "speed": 0.7},
        {"accuracy": 0.67, "speed": 0.6},
        {"accuracy": 0.68, "speed": 0.5},
        {"accuracy": 0.68, "speed": 0.4},
        {"accuracy": 0.68, "speed": 0.3},
    ]
    
    for i, scores in enumerate(simulated_scores):
        trial_id = f"trial_{i+1}"
        multi_eci.add_trial(trial_id, scores)
        
        print(f"   Trial {i+1}: Accuracy = {scores['accuracy']:.3f}, Speed = {scores['speed']:.3f}")
        
        if multi_eci.should_stop():
            print(f"   🛑 Multi-objective ECI detected convergence after {i+1} trials")
            break
    
    # Get multi-objective convergence information
    info = multi_eci.get_multi_objective_info()
    print("\n📊 Multi-Objective Convergence Analysis:")
    for key, value in info.items():
        if key == "objective_convergence":
            print(f"   {key}:")
            for obj, conv_info in value.items():
                print(f"     {obj}: {conv_info}")
        else:
            print(f"   {key}: {value}")
    
    return multi_eci


# ============================================================================
# STEP 5: Resource-Aware FLAML Example
# ============================================================================

def resource_aware_flaml_example():
    """Demonstrate resource-aware FLAML optimization."""
    
    print("\n💾 Resource-Aware FLAML Example")
    print("=" * 50)
    
    # Generate larger dataset to demonstrate resource constraints
    X, y = make_classification(
        n_samples=5000,
        n_features=50,
        n_informative=20,
        n_redundant=15,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"📊 Large dataset: {X.shape}")
    
    try:
        # Create resource-aware FLAML searcher
        resource_searcher = FLAMLResourceAwareSearcher(
            config_spaces={
                "random_forest": {
                    "n_estimators": {"type": "int", "bounds": [50, 200]},
                    "max_depth": {"type": "int", "bounds": [3, 10]}
                }
            },
            time_budget=60,  # 60 seconds
            memory_budget=1000  # 1GB memory budget
        )
        
        print("🔬 Running resource-aware FLAML optimization...")
        print("   Time budget: 60 seconds")
        print("   Memory budget: 1GB")
        
        start_time = time.time()
        
        # Run optimization
        best_config = resource_searcher.fit(X_train, y_train, task="classification")
        
        optimization_time = time.time() - start_time
        
        print(f"⏱️ Optimization completed in {optimization_time:.2f} seconds")
        print(f"🏆 Best configuration: {best_config}")
        
        # Get optimization summary
        summary = resource_searcher.get_optimization_summary()
        print(f"📊 Resource usage summary:")
        for key, value in summary.items():
            print(f"   {key}: {value}")
        
        return resource_searcher, best_config
        
    except ImportError as e:
        print(f"⚠️ FLAML not available: {e}")
        return None, None


# ============================================================================
# STEP 6: Factory Functions Example
# ============================================================================

def factory_functions_example():
    """Demonstrate the factory functions for creating searchers and ECIs."""
    
    print("\n🏭 Factory Functions Example")
    print("=" * 50)
    
    # Create searchers using factory functions
    print("🔧 Creating searchers with factory functions:")
    
    try:
        # Create FLAML searcher
        flaml_searcher = get_searcher(
            "flaml",
            config_spaces={
                "random_forest": {
                    "n_estimators": {"type": "int", "bounds": [50, 100]}
                }
            },
            time_budget=30
        )
        print("   ✅ FLAML searcher created")
        
        # Create Optuna searcher
        optuna_searcher = get_searcher(
            "optuna",
            config_spaces={
                "random_forest": {
                    "n_estimators": {"type": "int", "bounds": [50, 100]}
                }
            }
        )
        print("   ✅ Optuna searcher created")
        
    except ImportError as e:
        print(f"   ⚠️ Some searchers not available: {e}")
    
    # Create ECIs using factory functions
    print("\n🎯 Creating ECIs with factory functions:")
    
    standard_eci = get_eci("standard", window_size=5, patience=3)
    print("   ✅ Standard ECI created")
    
    adaptive_eci = get_eci("adaptive", window_size=5, patience=3)
    print("   ✅ Adaptive ECI created")
    
    multi_eci = get_eci("multi_objective", objectives=["accuracy", "speed"], window_size=5)
    print("   ✅ Multi-objective ECI created")
    
    return True


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("🚀 MLTeammate Advanced Search Tutorial")
    print("=" * 60)
    print("This tutorial demonstrates advanced search capabilities:")
    print("• FLAML-based hyperparameter optimization")
    print("• Early Convergence Indicators (ECI)")
    print("• Time-bounded optimization")
    print("• Resource-aware optimization")
    print("• Multi-objective optimization")
    print("=" * 60)
    
    # Step 1: Explore components
    explore_search_components()
    
    # Step 2: FLAML searcher
    flaml_searcher, flaml_config = flaml_searcher_example()
    
    # Step 3: ECI examples
    standard_eci, adaptive_eci = eci_example()
    
    # Step 4: Multi-objective ECI
    multi_eci = multi_objective_eci_example()
    
    # Step 5: Resource-aware FLAML
    resource_searcher, resource_config = resource_aware_flaml_example()
    
    # Step 6: Factory functions
    factory_functions_example()
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 ADVANCED SEARCH TUTORIAL COMPLETED!")
    print("=" * 60)
    print("✅ You've explored advanced search capabilities!")
    print("✅ FLAML integration for efficient optimization")
    print("✅ Early convergence detection to save time")
    print("✅ Resource-aware optimization for constrained environments")
    print("✅ Multi-objective optimization support")
    print("\n💡 Key Takeaways:")
    print("   • FLAML provides time-bounded optimization")
    print("   • ECI can save computational resources")
    print("   • Factory functions simplify component creation")
    print("   • Multiple optimization strategies available")
    print("\n🚀 Ready to use advanced search in your experiments!") 