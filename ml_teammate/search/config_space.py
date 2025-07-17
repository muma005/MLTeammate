# ml_teammate/search/config_space.py

lightgbm_config = {
    "max_depth": {"type": "int", "bounds": [3, 8]},
    "learning_rate": {"type": "float", "bounds": [0.01, 0.3]},
    "n_estimators": {"type": "int", "bounds": [50, 300]},
}

xgboost_config = {
    "max_depth": {"type": "int", "bounds": [3, 8]},
    "learning_rate": {"type": "float", "bounds": [0.01, 0.3]},
    "n_estimators": {"type": "int", "bounds": [50, 300]},
}
