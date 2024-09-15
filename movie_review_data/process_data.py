from utils import ProcessCSV
import os

def process_data():
    data_path = os.path.join(os.getcwd(), 'movie_review_data', 'data.csv')
    process = ProcessCSV(data_path)
    print(process.get_column())
    return process.get_df()

if __name__ == '__main__':
    print(process_data().head())
