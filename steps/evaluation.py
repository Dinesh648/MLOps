import logging
from zenml import step


@step
def evaluation(df: pd.DataFrame) -> None:
    """
    Evaluate the model on the ingested data

    Args:
        df: the ingested data

    Returns:
        None
    """
    pass