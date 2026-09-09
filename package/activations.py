from package.base import Activation, np

class ReLU(Activation):

    def __init__(self):
        self.input_cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input_cache = x
        return x * (x > 0)

    def backward(self, gradients: np.ndarray) -> np.ndarray:
        return gradients * (self.input_cache > 0)