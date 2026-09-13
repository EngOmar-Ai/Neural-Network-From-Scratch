import numpy as np

class NeuralNetwork:
    """
    A simple feedforward neural network composed of a sequence of layers.

    Layers are applied in order during the forward pass. Each layer is
    expected to implement a `forward` method (and typically a `backward`
    method, used by an optimizer such as StochasticGradientDescent).

    Attributes:
        layers (list | tuple): Ordered sequence of layer/activation objects
            making up the network.
    """

    def __init__(self, layers: list|tuple) -> None:
        """
        Initializes the network with a sequence of layers.

        Args:
            layers: Ordered collection of layer/activation objects, each
                exposing a `forward` method (and a `backward` method if the
                network will be trained).
        """

        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Runs a forward pass through all layers in sequence.

        Args:
           x: Input array, typically of shape (batch_size, in_features) for
               the first layer.

        Returns:
           Output array produced by passing x through every layer in order.
        """

        for layer in self.layers:
            x = layer.forward(x)
        return x