# README.md

## Project Overview
This project is a Python application designed for machine learning model training and inference. It leverages models in both Hugging Face and ONNX formats, enabling efficient training and inference processes. The application is modular, with each Python file serving a specific purpose.

## Files

### `__init__.py`
This file initializes the Python package and imports necessary functions from other files in the package. Specifically, the functions `download_save_raw_model` and `download_save_onnx_models` are imported from the `download_save_huggingface_model.py` file.

### `main.py`
This file contains the core logic of the application. It defines two primary classes:
- **Train**: Handles the training of models on specified datasets.
- **Inference**: Loads trained models and generates answers to input questions.

### `generate_onnx.py`
This file defines the `PhiONNXModelHandler` class, which manages ONNX models. It includes methods for:
- Loading the ONNX model.
- Creating a prompt.
- Generating answers based on the model.

### `base_generator.py`
This file defines the `HuggingFaceModelHandler` abstract base class. It provides a template for managing Hugging Face models, including:
- Abstract methods for loading models.
- Creating prompts.
- Generating answers from input questions.

### `models.py`
This file defines the `model_bindings` enum, which maps model names to their corresponding handler classes, ensuring models are handled correctly based on their type.

## Usage
To use this application, create an instance of the `Inference` class with the desired model name. You can then call the `generate_answer` method to generate an answer from the model. For example:

```python
Inference_obj = Inference("phi_onnx") 
Inference_obj.generate_answer("What is the capital of France?")
```

-> This will:
* Load the ONNX model named phi_onnx.
* Create a prompt using the question "What is the capital of France?".
* Generate and return an answer using the model.