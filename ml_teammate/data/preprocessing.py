from sklearn.preprocessing import StandardScaler, LabelEncoder

def scale_data(X_train, X_test):
    """
    Scale features using StandardScaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def encode_labels(y):
    """
    Encode target labels using LabelEncoder.
    """
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return y_encoded
