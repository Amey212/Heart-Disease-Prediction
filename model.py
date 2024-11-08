import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def train_and_save_model():
    data = pd.read_csv('heart(1).csv')
    X = data.drop('target', axis=1)
    y = data['target']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=4,
        class_weight={0: 1, 1: 3},
        random_state=42
    )
    rf_model.fit(X_train_scaled, y_train)

    # Evaluate the model
    y_pred = rf_model.predict(X_test_scaled)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save the model and scaler
    joblib.dump(rf_model, 'rf_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')

if __name__ == "__main__":
    train_and_save_model()