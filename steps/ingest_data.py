# steps to ingest the data to the pipeline
import logging
import pandas as pd
from zenml import step


class IngestData:
    """
    Ingest the data from the data path.
    """
    def __init__(self, data_path: str):
        """
        Args:
            data_path: path to the data
        """
        self.data_path = data_path

    def get_data(self):
        """
        Ingesting the data from the path into the pipeline
        """
        logging.info(f"Ingesting data from path {self.data_path}")
        return pd.read_csv(self.data_path)


@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    Ingest the data from the given path.

    Args:
        data_path:path to the data

        Returns:
            DataFrame: ingested data
    """
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.exception(f"Error ingesting data: {e}")
        raise e
