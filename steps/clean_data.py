import logging
import pandas as pd
from zenml import step

#to clean the data
@step
def clean_df(df: pd.DataFrame) -> None:
    pass
