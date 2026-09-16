from network.base import Layer
import numpy as np

class Linear(Layer):

    def __init__(self, in_features: int, out_features: int) -> None:

        self.weights = np.random.randn(in_features, out_features) * np.sqrt(2 / in_features)
        self.bias = np.zeros(out_features)

        self.input = None
        self.dw = None
        self.db = None

    def forward(self, x: np.ndarray, **kwargs) -> np.ndarray:

        self.input = x
        return (x @ self.weights) + self.bias

    def backward(self, gradients: np.ndarray, **kwargs) -> np.ndarray:

        batch_size = gradients.shape[0]

        self.dw = (self.input.T @ gradients) / batch_size
        self.db = np.sum(gradients, axis=0) / batch_size

        return gradients @ self.weights.T

    def parameters(self) -> list:
        return [(self.weights, self.dw), (self.bias, self.db)]

class ReLU(Layer):

    def __init__(self) -> None:

        self.input = None

    def forward(self, x: np.ndarray, **kwargs) -> np.ndarray:

        self.input = x
        return x * (x > 0)

    def backward(self, gradients: np.ndarray, **kwargs) -> np.ndarray:

        return gradients * (self.input > 0)

    def parameters(self) -> list:
        return []

class Dropout(Layer):

    def __init__(self, probability: float|int) -> None:

        if not (0 <= probability < 1):
            raise ValueError("Probability must be set between 0 and 1")

        self.probability = probability
        self.mask = None

    def forward(self, x: np.ndarray, **kwargs) -> np.ndarray:

        training = kwargs.get("training", False)

        if training:
            mask = (np.random.rand(*x.shape) > self.probability)
            self.mask = mask

            return (x * mask) / 1 - self.probability

        return x

    def backward(self, gradients: np.ndarray, **kwargs) -> np.ndarray:

        training = kwargs.get("training", False)

        if training:
             return gradients * self.mask

        return gradients

    def parameters(self) -> list:
        return []

if __name__ == "__main__":
    ...
