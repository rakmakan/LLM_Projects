import pandas as pd
from typing import List
# import logging
from utils.get_logger import get_logger


class ProcessCSV():

    def __init__(self, data_path) -> None:
        self.logger = get_logger()
        self.logger.info(f'Processing data from {data_path}')
        self.data_path = data_path
        self.df = pd.read_csv(self.data_path)

    def get_df(self) -> pd.DataFrame:
        return self.df

    def get_column(self) -> list:
        return self.df.columns