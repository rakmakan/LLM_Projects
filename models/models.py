from enum import Enum
from phi3B import PhiONNXModelHandler


class ModelClassBindings(Enum):
    """
    Enum class to represent the model bindings
    """
    phi_onnx = PhiONNXModelHandler