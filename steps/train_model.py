from zenml import step

from sklearn.ensemble import RandomForestClassifier

@step
def train_model(X_train, y_train):

    # Initialize the Logistic Regression model
    model = RandomForestClassifier(n_estimators=100, random_state=42)  # Increase max_iter if needed for convergence
    # Train the model on the training data
    model.fit(X_train, y_train)

    return model