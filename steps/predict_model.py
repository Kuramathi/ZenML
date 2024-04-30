from zenml import step
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report


@step
def predict_model(X_test, y_test, model) -> float:
    # Predict the labels for the test set
    y_pred = model.predict(X_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    return accuracy
