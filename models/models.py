from enum import Enum
from phi3B import PhiONNXModelHandler


class model_bindings(Enum):
    """
    Enum class to represent the model bindings
    """
    phi_onnx = PhiONNXModelHandler