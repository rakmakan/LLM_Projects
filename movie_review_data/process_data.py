from utils import ProcessCSV
from utils import get_logger    
import os

logger = get_logger()

def process_data():
    logger.info(f'Name of folder: movie_review_data')   
    data_path = os.path.join(os.getcwd(), 'movie_review_data', 'data.csv')
    process = ProcessCSV(data_path)
    process.get_df().columns = ['text', 'label']
    #log for columns
    logger.info(f'Columns: {process.get_column()}')
    return process.get_df()



if __name__ == '__main__':
    print(process_data().head())
