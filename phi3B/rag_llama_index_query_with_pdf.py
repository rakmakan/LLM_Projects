# main.py

import sys
import os
import json
from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,ServiceContext,PromptTemplate, get_response_synthesizer
from llama_index.core import GPTVectorStoreIndex, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import Settings
from phi3B import PhiONNXModelHandler
from llama_index.llms.huggingface import HuggingFaceLLM

from llama_index.core import StorageContext, load_index_from_storage
# Load configurations
with open('config.json', 'r') as f:
    config = json.load(f)

model_path = config.get('model_path', '/teamspace/studios/this_studio/models/phi_onnx/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4')
index_path = config.get('index_path', 'index_storage')

def build_index(pdf_path):
    # Load documents from the PDF
    documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()

    # Initialize the local embedding model
    embed_model = HuggingFaceEmbedding(model_name='all-MiniLM-L6-v2')

    # Create a ServiceContext with the custom embedding model
    Settings.embed_model = embed_model

    # Build the index

    index = VectorStoreIndex.from_documents(documents)

    # Save the index to disk
    index.storage_context.persist(persist_dir=index_path)
    print(f"Index built and saved to '{index_path}'.")

def interactive_query():
    # Initialize the PhiONNXModelHandler
    embed_model = HuggingFaceEmbedding(model_name='all-MiniLM-L6-v2')

    # Create a ServiceContext with the custom embedding model
    Settings.embed_model = embed_model
    phi_model_handler = PhiONNXModelHandler(verbose=True, timings=True)
    phi_model_handler.load_model(model_path)
    llm = HuggingFaceLLM(
        model_name="microsoft/Phi-3-mini-4k-instruct",
        model_kwargs={
            "trust_remote_code": True,
        })
    
    Settings.llm = llm
    # Load the index
    storage_context = StorageContext.from_defaults(persist_dir=index_path)
    index = load_index_from_storage(storage_context)

    # configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=1,
    )

    response_synthesizer = get_response_synthesizer()

   

    # Load the phi-ONNX model
    

   
    print("Welcome to the RAG system. Type 'exit' to quit.")
    while True:
        question = input("You: ")
        if question.lower() == 'exit':
            break

        # Retrieve context from the index
        # assemble query engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.4)],
        )

        context = query_engine.query(question)
        print(f"Context: {context}")

        # Generate a response using the phi-ONNX model
        prompt = phi_model_handler.create_prompt(f"{context}\n{question}")
        args = {
            'max_length': 100,
            'temperature': 0.7,
            'top_k': 10
        }
        answer = phi_model_handler.generate_answer(prompt, args)

        print(f"\nAssistant: {answer}")

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1].endswith('.pdf'):
        pdf_path = sys.argv[1]
        build_index(pdf_path)
    else:
        interactive_query()