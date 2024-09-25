# main.py

import sys
import os
import json
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader
from phi3B import PhiONNXModelHandler

# Load configurations
with open('config.json', 'r') as f:
    config = json.load(f)

model_path = config.get('model_path', 'models/your_phi_onnx_model.onnx')
index_path = config.get('index_path', 'index.json')

def build_index(pdf_path):
    # Load documents from the PDF
    documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()

    # Build the index
    index = GPTSimpleVectorIndex(documents)

    # Save the index to disk
    index.save_to_disk(index_path)
    print(f"Index built and saved to '{index_path}'.")

def interactive_query():
    # Load the index
    index = GPTSimpleVectorIndex.load_from_disk(index_path)

    # Initialize the PhiONNXModelHandler
    phi_model_handler = PhiONNXModelHandler(verbose=True, timings=True)

    # Load the phi-ONNX model
    phi_model_handler.load_model(model_path=model_path)

    print("Welcome to the RAG system. Type 'exit' to quit.")
    while True:
        question = input("You: ")
        if question.lower() == 'exit':
            break

        # Retrieve context from the index
        context = index.query(question)

        # Generate a response using the phi-ONNX model
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        args = {
            'max_length': 100,
            'temperature': 0.7,
            'top_k': 50
        }
        answer = phi_model_handler.generate_answer(prompt, args)

        print(f"\nAssistant: {answer}")

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1].endswith('.pdf'):
        pdf_path = sys.argv[1]
        build_index(pdf_path)
    else:
        interactive_query()
