import logging
import pandas as pd
from zenml import steps


@step
def train_model(df: pd.DataFrame) -> None:
    """
    Trains the model on the ingested data.

    Args:
        df (pd.DataFrame): The ingested data.

    Returns:
        None
    """
    pass
