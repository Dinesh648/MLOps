import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    """
    Abstract class for evaluation.
    """
    @abstractmethod
    def calculate_scores(self, y_true: pd.Series, y_pred: pd.Series):
        """
        Calculate the scores for the model
        Args:
            y_true: True labels
            y_pred: Predicted labels
        Returns:
            None
        """
        pass


class MSE(Evaluation):
    """
    Evaluation stategy that uses Mean Squared Error
    """
    def calculate_scores(self, y_true: pd.Series, y_pred: pd.Series):
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"MSE: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error calculating MSE: {e}")
            raise e


class R2(Evaluation):
    """
    Evaluation stategy that uses R2 Score
    """
    def calculate_scores(self, y_true: pd.Series, y_pred: pd.Series):
        try:
            logging.info("Calculating R2")
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R2: {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error calculating R2: {e}")
            raise e


class RMSE(Evaluation):
    """
    Evaluation stategy that uses Root Mean Squared Error
    """
    def calculate_scores(self, y_true: pd.Series, y_pred: pd.Series):
        try:
            logging.info("Calculating RMSE")
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info(f"RMSE: {rmse}")
            return rmse
        except Exception as e:
            logging.error(f"Error calculating RMSE: {e}")
            raise e
