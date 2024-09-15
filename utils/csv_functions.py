import pandas as pd

class ProcessCSV():

    def __init__(self, data_path) -> None:
        self.data_path = data_path
        self.df = pd.read_csv(self.data_path)

    def get_df(self) -> pd.DataFrame:
        return self.df

    def get_column(self) -> list:
        return self.df.columns