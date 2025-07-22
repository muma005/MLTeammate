from .xgboost_learner import XGBoostLearner

def get_learner(name):
    if name == "xgboost":
        return XGBoostLearner
    # Add more learners as needed
    raise ValueError(f"Unknown learner: {name}")