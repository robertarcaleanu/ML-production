import logging
import pandas as pd
from abc import ABC, abstractmethod

from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Abstract clss for all models
    """
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        pass

class LinearRegressionModel(Model):
    """
    Linear regression model.
    """
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model training completed")
            return reg
        except Exception as e:
            logging.error("Error in training model: {}".format(e))
            raise e