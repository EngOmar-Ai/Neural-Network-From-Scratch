from package.base import Layer, np

class Linear(Layer):
    """
    Fully connected (dense) layer applying a linear transformation y = xW + b.

    Weights are initialized using He initialization (scaled by sqrt(2 / in_features)),
    which works well for layers followed by ReLU-family activations. Biases are
    initialized to zero.

    Attributes:
        weights (np.ndarray): Weight matrix of shape (in_features, out_features).
        bias (np.ndarray): Bias vector of shape (out_features,).
        input_cache (np.ndarray | None): Input from the most recent forward pass,
            cached for use in backpropagation.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        """
        Fully connected (dense) layer applying a linear transformation y = xW + b.

        Weights are initialized using He initialization (scaled by sqrt(2 / in_features)),
        which works well for layers followed by ReLU-family activations. Biases are
        initialized to zero.

        Attributes:
            in_features (np.ndarray): The size of the input features the layer will receive.
            out_features (np.ndarray): the size of the output features the layer will produce.
        """

        self.weights = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.bias = np.zeros((out_features,))

        self.input_cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass of the linear layer.

        Caches the input for use during the backward pass.

        Args:
            x: Input array of shape (batch_size, in_features).

        Returns:
            Output array of shape (batch_size, out_features), computed as
            (x @ weights) + bias.
        """

        self.input_cache = x
        return (x @ self.weights) + self.bias

    def backward(self, gradients: np.ndarray, learning_rate: float|int) -> np.ndarray:
        """
        Computes the backward pass and updates the layer's parameters in place.

        Computes gradients of the loss with respect to the weights and bias
        (averaged over the batch), applies a vanilla SGD update to both, and
        propagates the gradient back to the previous layer.

        Args:
            gradients: Gradient of the loss with respect to this layer's output,
                of shape (batch_size, out_features).
            learning_rate: Step size used for the SGD parameter update.

        Returns:
            Gradient of the loss with respect to this layer's input, of shape
            (batch_size, in_features), to be passed to the previous layer.
        """

        batch_size = gradients.shape[0]

        dw = (self.input_cache.T @ gradients) / batch_size
        db = np.sum(gradients, axis=0) / batch_size

        next_layer_gradients = gradients @ self.weights.T

        self.weights -= dw * learning_rate
        self.bias -= db * learning_rate

        return next_layer_gradients