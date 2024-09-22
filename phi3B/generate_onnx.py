from abc import ABC, abstractmethod
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
import torch
from models import HuggingFaceModelHandler
import os
class PhiONNXModelHandler(HuggingFaceModelHandler):
    def __init__(self, device=None):
        """
        Initialize the ONNX model handler by downloading the model and tokenizer.
        """
        super().__init__(device)
        self.past_key_values = None  # Initialize past key values as None

    def load_model(self, model_path):
        """
        Load the ONNX model and tokenizer from Hugging Face.
        """

        self.option = ort.SessionOptions()
        self.option.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        # Load the ONNX model using ONNX Runtime
        self.ort_session = ort.InferenceSession(model_path, self.option, providers=["CPUExecutionProvider"]) 

        # Load the tokenizer for the model dir path
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(model_path))

    def create_prompt(self, question: str, template: str = "Q: {question}\nA:"):
        """
        Create a formatted prompt using a question and a template format is as follows:
        <|system|>
        You are a helpful assistant.<|end|>
        <|user|>
        How to explain Internet for a medieval knight?<|end|>
        <|assistant|>
        """
        template = "<|system|>\nYou are a helpful assistant.<|end|>\n<|user|>\n{question}<|end|>\n<|assistant|>"
        return template.format(question=question)

    def initialize_past_key_values(self, batch_size: int, num_layers: int, seq_length: int, head_size: int):
        """
        Initialize past key values as zero tensors for the first inference.
        """
        # Initialize past_key_values as zero tensors for the first run
        return [
            (np.zeros((batch_size, 32, seq_length, head_size), dtype=np.float32),
             np.zeros((batch_size, 32, seq_length, head_size), dtype=np.float32))
            for _ in range(num_layers)
        ]



    def generate_answer(self, prompt: str, max_length=4096, temperature=0.1):
        """
        Generate an answer based on the prompt using the ONNX model.
        """
        # Tokenize the input prompt
        inputs = self.tokenizer(prompt, return_tensors="np")

        # Extract input_ids and attention_mask from the tokenizer output
        input_ids = inputs['input_ids'].astype(np.int64)
        attention_mask = inputs['attention_mask'].astype(np.int64)

        # Initialize past_key_values with zeros for the first pass if None
        if self.past_key_values is None:
            batch_size = input_ids.shape[0]
            num_layers = 32  # From the config, we have 32 layers
            seq_length = input_ids.shape[1]
            head_size = 96  # Head size from the config
            self.past_key_values = self.initialize_past_key_values(batch_size, num_layers, seq_length, head_size)

        # Prepare ONNX model inputs
        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        # Pass the initialized past_key_values
        for i, (key, value) in enumerate(self.past_key_values):
            ort_inputs[f"past_key_values.{i}.key"] = key
            ort_inputs[f"past_key_values.{i}.value"] = value

        # Run the ONNX model to generate output
        ort_outputs = self.ort_session.run(None, ort_inputs)
        # Extract logits (predictions) from the output (assuming logits are the first output)
        logits = ort_outputs[0]

        # Convert logits to token IDs by taking the argmax (most likely token) along the last dimension
        token_ids = np.argmax(logits, axis=-1)

        # Store past_key_values for future predictions
        num_past_layers = (len(ort_outputs) - 1) // 2
        self.past_key_values = [
            (ort_outputs[i + 1], ort_outputs[i + 2]) for i in range(num_past_layers)
        ]

        # Flatten the list of generated token IDs (if needed)
        flat_generated_tokens = token_ids.flatten().tolist()

        # Decode the generated tokens to text
        generated_text = self.tokenizer.decode(flat_generated_tokens, skip_special_tokens=True)

        return generated_text




# Example usage
if __name__ == "__main__":
    # Instantiate the ONNX model handler
    onnx_handler = PhiONNXModelHandler(device="cpu")

    # Load the model
    onnx_handler.load_model()

    # Create a prompt
    question = "What is the capital of France?"
    prompt = onnx_handler.create_prompt(question)

    # Generate an answer
    answer = onnx_handler.generate_answer(prompt)
    print(answer)
