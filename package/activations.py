from package.base import Activation, np

class ReLU(Activation):
    """
    Rectified Linear Unit activation function.

   Applies the elementwise transformation f(x) = max(0, x) during the forward
   pass. This is a parameter-free activation, so backpropagation only
   computes the gradient with respect to the input.

   Attributes:
       input_cache (np.ndarray | None): Input from the most recent forward pass,
           cached for use in backpropagation.
   """

    def __init__(self) -> None:
        """Initializes the input cache to None."""

        self.input_cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass of the ReLU activation.

        Caches the input for use during the backward pass.

        Args:
            x: Input array of any shape.

        Returns:
            Array of the same shape as x, with negative values zeroed out
            (elementwise max(0, x)).
        """

        self.input_cache = x
        return x * (x > 0)

    def backward(self, gradients: np.ndarray) -> np.ndarray:
        """
        Computes the backward pass of the ReLU activation.

        Propagates the incoming gradient only through positions where the
        cached input was positive, zeroing it out elsewhere (the gradient of
        ReLU is 1 where input > 0, and 0 otherwise).

        Args:
            gradients: Gradient of the loss with respect to this layer's
                output, same shape as the cached input.

        Returns:
            Gradient of the loss with respect to this layer's input, same
            shape as gradients.
        """
        return gradients * (self.input_cache > 0)