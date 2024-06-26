import logging
import pandas as pd

# from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
# rom .config import ModelNameConfig
from typing import Literal

# @step
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    model_name: Literal["LinearRegression"],
) -> RegressorMixin:
    """Train the model on the ingested data

    Args:
        X_train (pd.DataFrame): train data
        X_test (pd.DataFrame): test data
        y_train (pd.DataFrame): train value
        y_test (pd.DataFrame): test value
    """
    try:
        model = None
        if model_name == "LinearRegression":
            model = LinearRegressionModel()
            train_model = model.train(X_train, y_train)
            return train_model
        else:
            raise ValueError("Model {} not supported".format(model_name))
    except Exception as e:
        logging.error("Error in training the model: {}".format(model_name))
        raise e