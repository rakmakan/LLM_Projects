import movie_review_data
from utils import get_logger
import json
from sklearn.model_selection import train_test_split
from utils import download_save_raw_model, download_save_onnx_models
from models import ModelClassBindings
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


class Inference_LLM():
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
        handler = ModelClassBindings[self.model_name].value()
        handler.load_model(model_path)
        prompt = handler.create_prompt(question)
        answer = handler.generate_answer(prompt, self.config['args'])

    def chat_mode(self):
        handler = ModelClassBindings[self.model_name].value()
        handler.load_model(self.config['model_path'])
        handler.chat_mode(self.config['args'])

if __name__ == '__main__':
    Inference_obj = Inference_LLM("phi_onnx") 
    Inference_obj.generate_answer("""
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    left = []
    right = []
<|fim_hole|>
    return quick_sort(left) + [pivot] + quick_sort(right)
""")
    # Inference_obj.chat_mode()
    

