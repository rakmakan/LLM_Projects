
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.base_generator import HuggingFaceModelHandler   

class DeepSeekCodeHandler(HuggingFaceModelHandler):
    def __init__(self, model_name='deepseek-ai/deepseek-coder-6.7b-instruct', device=None):
        """
        Initialize the DeepSeekCodeHandler with the specific DeepSeek model.
        """
        super().__init__(device)

    def load_model(self, model_path = None):
        """
        Load the DeepSeek model and tokenizer.
        """
    
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(self.device)

    def create_prompt(self, question: str, template: str = "Q: {question}\nA:"):
        """
        Create a prompt by formatting the input question according to a template.
        """
        return template.format(question=question)

    def handle_code(self, input_code, max_length=128):
        """
        Handle both code completion and code insertion based on whether the input contains a placeholder.
        """
        # Check if the input_code contains the placeholder for code insertion
        if '<|fim_hole|>' in input_code:
            print("Performing code insertion...")
            inputs = self.tokenizer(input_code, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, max_length=max_length)
            inserted_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)[len(input_code):]
            return inserted_code
        else:
            print("Performing code completion...")
            inputs = self.tokenizer(input_code, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, max_length=max_length)
            completed_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return completed_code

    def generate_answer(self, query: str, args):
        """
        Handle general questions, code completion, or code insertion based on the query input.
        """
        # Check if the query is asking for code completion or insertion
        max_length = args.get('max_length', 128)
        temperature = args.get('temperature', 0.7)

        if 'def ' in query or 'class ' in query or 'import ' in query:
            # Treat it as a code-related query (completion or insertion)
            print("Detected code-related query. Handling as code completion or insertion...")
            return self.handle_code(query, max_length)
        else:
            # Treat it as a general question for the chatbot
            print("Handling as a general question...")
            prompt = self.create_prompt(query)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                inputs['input_ids'], max_length=max_length, temperature=temperature)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)