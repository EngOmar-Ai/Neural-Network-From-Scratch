from package.base import Layer
import numpy as np

class Linear(Layer):

    def __init__(self, in_features: int, out_features: int) -> None:
        ...

    def forward(self, x: np.ndarray, **kwargs) -> np.ndarray:
        ...

    def backward(self, gradients: np.ndarray, **kwargs) -> np.ndarray:
        ...

class ReLU(Layer):

    def __init__(self) -> None:
        ...

    def forward(self, x: np.ndarray, **kwargs) -> np.ndarray:
        ...

    def backward(self, gradients: np.ndarray, **kwargs) -> np.ndarray:
        ...

class Dropout(Layer):

    def __init__(self, probability) -> None:
        ...

    def forward(self, x: np.ndarray, **kwargs) -> np.ndarray:
        ...

    def backward(self, gradients: np.ndarray, **kwargs) -> np.ndarray:
        ...

if __name__ == "__main__":
    ...