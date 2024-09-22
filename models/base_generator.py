from abc import ABC, abstractmethod
import torch

class HuggingFaceModelHandler(ABC):
    def __init__(self, device=None):
        """
        Initialize the Hugging Face model handler by downloading the model and tokenizer.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def load_model(self):
        """
        Abstract method to load the model and tokenizer.
        """
        pass

    @abstractmethod
    def create_prompt(self, question: str, template: str = "Q: {question}\nA:"):
        """
        Abstract method to create a prompt for the model.
        """
        pass

    @abstractmethod
    def generate_answer(self, prompt: str, max_length=100, temperature=0.7):
        """
        Abstract method to generate an answer based on the prompt.
        """
        pass