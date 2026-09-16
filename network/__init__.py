import numpy as np

class NeuralNetwork:

    def __init__(self, layers: list|tuple) -> None:
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss_gradients: np.ndarray) -> None:
        for layer in reversed(self.layers):
            loss_gradients = layer.backward(loss_gradients)

    def parameters(self):
        params = []
        for layer in self.layers:
            params.append(layer.parameters())
        return params

if __name__ == "__main__":
    ...