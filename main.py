import movie_review_data
from utils import get_logger
#test and train
from sklearn.model_selection import train_test_split
from phi3B import PhiONNXModelHandler
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

    def generate_answer(self, question):
        onnx_handler = PhiONNXModelHandler(device="cpu")

        model_path = "/Users/rakshitmakan/Documents/projects/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/phi3-mini-4k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx"
        # Load the model
        onnx_handler.load_model(model_path=model_path)

        # Create a prompt
        question = "What is the capital of France?"
        prompt = onnx_handler.create_prompt(question)

        # Generate an answer
        answer = onnx_handler.generate_answer(prompt)
        return answer

if __name__ == '__main__':
    train = Train('movie_review_data')
    print(train.generate_answer('What is the capital of India?'))

