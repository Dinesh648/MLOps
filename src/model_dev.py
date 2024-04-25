import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Abstract class for all models.
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model on the given data.
        Args:
            X_train: Training data.
            y_train: Training labels.
        Returns:
            None
        """
        pass

class LinearRegressionModel(Model):
    """
    Linear Regression Model
    """
    def train(self, X_train, y_train, **kwargs):
        """
        Trains the Linear Regression model on the given data.
        Args:
            X_train: Training data.
            y_train: Training labels.
            **kwargs: Additional keyword arguments.
        Returns:
            None
        """
        try:
            reg = LinearRegression(**kwargs)
            print(X_train.isnull().sum())
            reg.fit(X_train, y_train)
            logging.info("Linear Regression model trained.")
            return reg
        except Exception as e:
            logging.error(f"Error training Linear Regression model: {e}")
            raise e
