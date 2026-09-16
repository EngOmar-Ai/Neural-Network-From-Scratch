from network.base import Optimizer
from network import NeuralNetwork

import numpy as np

class StochasticGradientDescent(Optimizer):

    def __init__(self, neural_network: NeuralNetwork, learning_rate: float|int) -> None:

        self.neural_network = neural_network
        self.learning_rate = learning_rate

        self.steps = 0

    def step(self, **kwargs) -> None:

        for parameters, gradients in self.neural_network.parameters():
            parameters -= self.learning_rate * gradients

        self.steps = self.steps + 1

class Adam(Optimizer):
    ...

if __name__ == "__main__":
    ...