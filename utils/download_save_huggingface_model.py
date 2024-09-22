# save and download huggingface model

import os
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.get_logger import get_logger 
import subprocess
logger = get_logger()

def download_save_raw_model(model_name, save_path):
    # download only if the model is not already downloaded
    if not os.path.exists(save_path):
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        logger.info(f"Model saved at: {save_path}")
    else:
        logger.info(f"Model already exists at: {save_path}")
        

def download_save_onnx_models(model_name, save_path, opset_version=11):
    '''
        download and save only if the model is not already downloaded using subprocess
        example: python -m transformers.onnx --model=distilbert/distilbert-base-uncased onnx/

    '''
    logger.info(f"Downloading model: {model_name}")
    if not os.path.exists(save_path):
        cmd = f'python -m transformers.onnx --model={model_name} {save_path}'
        subprocess.run(cmd, shell=True)
        logger.info(f"Model saved at: {save_path}")
    else:
        logger.info(f"Model already exists at: {save_path}")


