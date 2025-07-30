# docs/fixes.md
## Cross-Validation Fix (2025-07-30)
- Fixed parameter passing in XGBoostLearner
- Now handles both:
  ```python
  XGBoostLearner({"n_estimators": 100})  # Original style
  XGBoostLearner(n_estimators=100)       # New sklearn-compatible style

  
2. **Update Test Cases**
```python
# tests/test_learners.py
def test_parameter_styles():
    # Test both initialization methods
    model1 = XGBoostLearner({"max_depth": 5})
    model2 = XGBoostLearner(max_depth=5)
    assert model1.get_params()["max_depth"] == 5
    assert model2.get_params()["max_depth"] == 5