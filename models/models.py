from enum import Enum
from phi3B import PhiONNXModelHandler
from DeepSeekCoder import DeepSeekCodeHandler

class ModelClassBindings(Enum):
    """
    Enum class to represent the model bindings
    """
    phi_onnx = PhiONNXModelHandler
    deep_seek = DeepSeekCodeHandler