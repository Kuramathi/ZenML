from zenml import pipeline

from steps.data_loader import data_loader
from steps.data_preprocessor import data_preprocessor
from steps.split_data import split_data
from steps.train_model import train_model
from steps.predict_model import predict_model

@pipeline
def pipeline1():
    df_tuples = data_loader(base_path='../data')
    print("OK")
    df_preprocessed = data_preprocessor(df_tuples)
    print("OK")
    X_train, X_test, y_train, y_test = split_data(df_preprocessed)
    print("OK")
    model = train_model(X_train, y_train)
    print("OK")
    report = predict_model(X_test, y_test, model)
    print("OK")

    return report