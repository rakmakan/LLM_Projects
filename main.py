import movie_review_data
from utils import get_logger
import json
from sklearn.model_selection import train_test_split
from utils import download_save_raw_model, download_save_onnx_models
from models import model_bindings
logger = get_logger()


class Train():
    def __init__(self, data_name):
        self.data_name = data_name
        logger.info(f'Training data: {self.data_name}')
        self.split_data(0.2)

    
    def split_data(self, test_size):
        logger.info(f'Splitting data into train and test with test size: {test_size}')
        self.df = movie_review_data.process_data()
        logger.info(f'Dataframe shape: {self.df.shape}')
        self.X_train, self.y_train, self.X_text, self.y_test = train_test_split(self.df['text'], self.df['label'], test_size=test_size, random_state=42)


class Inference():
    def __init__(self, model_name):
        logger.info('Inference class initialized')
        self.model_name = model_name
        self.config = self.get_config()

    def get_config(self):
        with open('config.json') as f:
            config = json.load(f)
        return config[self.model_name]

    def generate_answer(self, question):   

        model_path = self.config['model_path']
        handler = model_bindings[self.model_name].value()
        handler.load_model(model_path)
        prompt = handler.create_prompt(question)
        answer = handler.generate_answer(prompt, self.config['args'])

if __name__ == '__main__':
    Inference_obj = Inference("phi_onnx") 
    Inference_obj.generate_answer("What is the capital of France?")
    

