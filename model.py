import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def train_and_save_model():
    data = pd.read_csv('heart(1).csv')
    X = data.drop('target', axis=1)
    y = data['target']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features (especially important for neural networks)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the MLP model
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(100, 50),  # Two hidden layers with 100 and 50 neurons respectively
        activation='relu',            # Rectified Linear Unit activation
        solver='adam',                # Optimizer
        alpha=0.0001,                # L2 regularization term
        batch_size='auto',           # Automatic minibatch size
        learning_rate='adaptive',     # Learning rate adjusts during training
        max_iter=500,                 # Maximum number of iterations
        random_state=42,
        early_stopping=True,         # Stop if validation score isn't improving
        validation_fraction=0.1       # Fraction of training data for validation
    )
    
    mlp_model.fit(X_train_scaled, y_train)

    # Evaluate the model
    y_pred = mlp_model.predict(X_test_scaled)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save the model and scaler
    joblib.dump(mlp_model, 'mlp_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')

    return mlp_model

if __name__ == "__main__":
    trained_model = train_and_save_model()