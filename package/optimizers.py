from package.network import NeuralNetwork, np
from package.base import Optimizer

class StochasticGradientDescent(Optimizer):

    def __init__(self, neural_network: NeuralNetwork, learning_rate: float|int) -> None:
        self.neural_network = neural_network
        self.learning_rate = learning_rate

    def step(self, loss_gradients: np.ndarray) -> None:
        for layer in reversed(self.neural_network.layers):
            loss_gradients = layer.backward(loss_gradients, self.learning_rate)