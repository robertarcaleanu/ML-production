import logging
import pandas as pd

import mlflow
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

# @step(experiment_tracker=experiment_tracker.name)
@step
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig,
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
        if config.model_name == "LinearRegression":
            mlflow.sklearn.autolog()  # To get loads
            model = LinearRegressionModel()
            train_model = model.train(X_train, y_train)
            return train_model
        else:
            raise ValueError("Model {} not supported".format(config.model_name))
    except Exception as e:
        logging.error("Error in training the model: {}".format(config.model_name))
        raise e