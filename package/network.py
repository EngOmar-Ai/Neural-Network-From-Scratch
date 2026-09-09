import numpy as np

class NeuralNetwork:

    def __init__(self, layers: list|tuple):
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x