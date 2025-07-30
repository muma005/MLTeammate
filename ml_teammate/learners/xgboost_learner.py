#learners/xgboost_learner.py
# ml_teammate/learners/xgboost_learner.py
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

class XGBoostLearner(BaseEstimator, ClassifierMixin):
    def __init__(self, config=None, **kwargs):  # NEW: Added kwargs capture
        """
        Now accepts both:
        - XGBoostLearner({"n_estimators": 100})
        - XGBoostLearner(n_estimators=100)
        """
        self.config = (config or {}).copy()  # Ensure we don't modify input
        self.config.update(kwargs)  # NEW: Merge keyword arguments
        self.model = None
        
        # Immediate initialization if params provided
        if self.config:
            self.model = XGBClassifier(**self.config)

    def fit(self, X, y):
        if self.model is None:
            self.model = XGBClassifier(**self.config)
        self.model.fit(X, y)
        return self

    # ... (rest of the methods remain unchanged)

    # EXISTING METHODS (unchanged)
    def predict(self, X):
        return self.model.predict(X)

    # NEW METHODS (Phase 2 additions)
    def get_params(self, deep=True):  # Required by sklearn
        return self.config.copy()  # Isolates config

    def set_params(self, **params):  # Required by sklearn
        self.config.update(params)
        return self  # Chainable

# Preserve existing factory function (Principle 1: No breaking changes)
def get_xgboost_learner(config):  
    return XGBoostLearner(config)