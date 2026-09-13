from package.network import NeuralNetwork, np
from package.base import Optimizer, Layer, Activation

class StochasticGradientDescent(Optimizer):
    """
    Vanilla (mini-batch) stochastic gradient descent optimizer.

    Drives backpropagation through a NeuralNetwork's layers in reverse order,
    passing the learning rate down to each layer so it can update its own
    parameters during its `backward` call.

    Attributes:
        neural_network (NeuralNetwork): The network whose layers will be
            updated.
        learning_rate (float | int): Step size used for parameter updates.
    """

    def __init__(self, neural_network: NeuralNetwork, learning_rate: float|int) -> None:
        """
        Initializes the optimizer.

        Args:
            neural_network: The network to optimize.
            learning_rate: Step size applied to parameter updates during
                each `step` call.
        """

        self.neural_network = neural_network
        self.learning_rate = learning_rate

    def step(self, loss_gradients: np.ndarray) -> None:
        """Performs one backward pass through the network, updating parameters.

        Iterates over the network's layers in reverse, propagating gradients
        from the output layer back to the input. `Layer` instances receive
        `self.learning_rate` so they can update their own parameters;
        `Activation` instances are parameter-free and are called without it.

        Args:
            loss_gradients: Gradient of the loss with respect to the
                network's final output, of shape matching that output.

        Returns:
            None. Parameters are updated in place on each layer.

        Raises:
            TypeError: If a layer in the network is neither an `Activation`
                nor a `Layer`.
        """

        for layer in reversed(self.neural_network.layers):
            if isinstance(layer, Activation):
                loss_gradients = layer.backward(loss_gradients)
            elif isinstance(layer, Layer):
                loss_gradients = layer.backward(loss_gradients, self.learning_rate)
            else:
                raise TypeError(f"Expected an Activation Function or a Neural Layer, Received {type(layer)}")

class Adam(Optimizer):
    raise NotImplementedError