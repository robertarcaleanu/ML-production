import logging
import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, r2_score, rm

class Evaluation(ABC):
    """Abstract class defining strategy for evaluating our models"""
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Calculates the score of the model."""
        pass

class MSE(Evaluation):
    """Evaluation strategy that uses Mean Squared Error."""
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info("MSE {}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error in calculating MSE: {}".format(e))
            raise e
        
class R2(Evaluation):
    """Evaluation strategy that uses R²"""
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating R2 Score")
            r2 = r2_score(y_true, y_pred)
            logging.info("R2 {}".format(r2))
            return r2
        except Exception as e:
            logging.error("Error in calculating MSE: {}".format(e))
            raise e
        

class RMSE(Evaluation):
    """Evaluation strategy that uses Root Mean Squared Error."""
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating RMSE")
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            logging.info("MSE {}".format(rmse))
            return rmse
        except Exception as e:
            logging.error("Error in calculating MSE: {}".format(e))
            raise e