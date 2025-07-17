from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def oversample_smote(X, y):
    """
    Apply SMOTE over-sampling to balance classes.
    """
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res

def undersample_random(X, y):
    """
    Apply random under-sampling to balance classes.
    """
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X, y)
    return X_res, y_res
