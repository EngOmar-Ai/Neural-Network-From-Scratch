from package.base import Optimizer
from package.network import NeuralNetwork
import numpy as np

class StochasticGradientDescent(Optimizer):

    def __init__(self, neural_network: NeuralNetwork, learning_rate: float|int) -> None:
        ...

    def step(self, loss_gradients: np.ndarray, **kwargs) -> None:
        ...

class Adam(Optimizer):
    ...