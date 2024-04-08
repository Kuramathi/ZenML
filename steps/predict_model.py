from zenml import step
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report


def predict_model(X_test, y_test, model):
    # Predict the labels for the test set
    y_pred = model.predict(X_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Generate a classification report
    report = classification_report(y_test, y_pred, target_names=df['label'].astype('category').cat.categories)
    print(report)

    return report