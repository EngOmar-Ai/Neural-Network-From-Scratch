from network.base import Optimizer
from network import NeuralNetwork

import numpy as np

class StochasticGradientDescent(Optimizer):

    def __init__(self, neural_network: NeuralNetwork, learning_rate: float|int) -> None:

        self.neural_network = neural_network
        self.learning_rate = learning_rate

        self.step = 0

    def step(self, loss_gradients: np.ndarray, **kwargs) -> None:

        for parameters, gradients in self.neural_network.parameters():
            parameters = parameters - (self.learning_rate * gradients)

        self.step = self.step + 1

class Adam(Optimizer):
    ...