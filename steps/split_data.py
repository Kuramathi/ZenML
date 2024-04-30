from zenml import step
import pandas as pd
import numpy as np
from typing import Tuple

from sklearn.model_selection import train_test_split


@step
def split_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Convert the features into a numpy array and stack them into a matrix
    X = np.stack(df['features'].values)

    # Encode the labels as integers
    y = df['label'].astype('category').cat.codes

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
