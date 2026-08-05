import numpy as np

class NeuralNetwork:
    """A simple feedforward neural network composed of a sequence of layers.

    Layers are applied in order during the forward pass and in reverse
    order during the backward pass, following the standard backpropagation
    pattern. Each layer is expected to implement its own `forward` and
    `backward` methods.
    """

    def __init__(self, layers: list | tuple, learning_rate: float | int):
        """
        Args:
            layers: An ordered collection of layer objects (e.g. Linear, ReLU)
                that make up the network, applied in the given order.
            learning_rate: The step size used by each layer's parameter
                update during backpropagation.
        """

        self.layers = layers
        self.learning_rate = learning_rate

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run the input through every layer in order.

        Args:
            x: Input batch, shape (batch_size, in_features).

        Returns:
            The network's output (e.g. logits), shape (batch_size, out_features).
        """

        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y: np.ndarray) -> None:
        """Propagate the loss gradient backward through every layer in reverse order.

        Each layer computes its own parameter gradients, updates its
        parameters in place, and returns the gradient with respect to its
        input, which is passed on to the preceding layer.

        Args:
            y: Gradient of the loss with respect to the network's output,
                shape (batch_size, out_features).
        """

        for layer in reversed(self.layers):
            y = layer.backward(y, self.learning_rate)


class Linear:
    """A fully connected (dense) layer: output = x @ weights + bias.

    Weights are initialized using He initialization, suitable for use with
    ReLU-family activations.
    """

    def __init__(self, in_features, out_features):
        """
        Args:
            in_features: Number of input features per sample.
            out_features: Number of output features per sample.
        """

        self.weights = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.bias = np.zeros((1, out_features))

        self.input_cache = None

    def forward(self, x: np.ndarray):
        """Compute the linear transformation and cache the input for backward.

        Args:
            x: Input batch, shape (batch_size, in_features).

        Returns:
            Output batch, shape (batch_size, out_features).
        """

        self.input_cache = x
        return (x @ self.weights) + self.bias

    def backward(self, y: np.ndarray, learning_rate: float | int):
        """Compute gradients, update parameters, and return the input gradient.

        Uses the input cached during the forward pass to compute the weight
        and bias gradients,

        updates `weights` and `bias` in place via
        gradient descent, and returns the gradient with respect to this
        layer's input (computed using the pre-update weights) for use by
        the previous layer.

        Args:
            y: Gradient of the loss with respect to this layer's output,
                shape (batch_size, out_features).
            learning_rate: Step size for the parameter update.

        Returns:
            Gradient of the loss with respect to this layer's input,
            shape (batch_size, in_features).
        """

        batch_size = self.input_cache.shape[0]

        dw = (self.input_cache.T @ y) / batch_size
        db = (np.sum(y, axis=0)) / batch_size

        output = y @ self.weights.T

        self.weights -= learning_rate * dw
        self.bias -= learning_rate * db

        return output


class ReLU:
    """Elementwise Rectified Linear Unit activation: f(x) = max(0, x)."""

    def __init__(self):
        self.input_cache = None

    def forward(self, x: np.ndarray):
        """Apply ReLU and cache the input for backward.

        Args:
            x: Input batch of any shape.

        Returns:
            Elementwise max(0, x), same shape as input.
        """

        self.input_cache = x
        return x * (x > 0)

    def backward(self, y: np.ndarray, learning_rate: float | int):
        """Backpropagate through ReLU using the cached forward input.

        Zeroes out gradients at positions where the original forward input
        was non-positive, and passes gradients through unchanged elsewhere.

        Args:
            y: Gradient of the loss with respect to this layer's output.
            learning_rate: Unused (ReLU has no parameters), kept for
                interface consistency with other layers.

        Returns:
            Gradient of the loss with respect to this layer's input.
        """

        return y * (self.input_cache > 0)


class MeanSquareError:
    """Mean squared error loss, averaged over features and batch."""

    @staticmethod
    def forward(prediction: np.ndarray, label: np.ndarray):
        """Compute the mean squared error between predictions and labels.

        Args:
            prediction: Predicted values, shape (batch_size, n_features).
            label: Target values, same shape as prediction.

        Returns:
            Scalar loss, averaged over both the feature and batch dimensions.
        """

        return ((prediction - label) ** 2).mean(axis=1).mean()

    @staticmethod
    def backward(prediction: np.ndarray, label: np.ndarray):
        """Compute the gradient of MSE with respect to the prediction.

        Args:
            prediction: Predicted values, shape (batch_size, n_features).
            label: Target values, same shape as prediction.

        Returns:
            Gradient of the loss with respect to prediction, same shape as
            prediction.
        """

        return 2 * (prediction - label) / prediction.shape[-1]


class CrossEntropyLoss:
    """Softmax combined with categorical cross-entropy loss.

    Expects raw, unnormalized logits as `prediction` and applies softmax
    internally. Fusing softmax and cross-entropy this way gives a simple,
    numerically stable gradient (`probabilities - label`) with respect to
    the logits, avoiding the need to backpropagate through softmax's full
    Jacobian separately.
    """

    @staticmethod
    def forward(prediction: np.ndarray, label: np.ndarray):
        """Compute the mean cross-entropy loss over the batch.

        Args:
            prediction: Raw logits (pre-softmax), shape (batch_size, n_classes).
            label: One-hot encoded targets, same shape as prediction.

        Returns:
            Scalar loss, averaged over the batch.
        """

        shifted = prediction - np.max(prediction, axis=-1, keepdims=True)
        exponent = np.exp(shifted)
        probabilities = exponent / np.sum(exponent, axis=-1, keepdims=True)
        loss = -np.sum(label * np.log(probabilities + 1e-9), axis=-1)
        return loss.mean()

    @staticmethod
    def backward(prediction: np.ndarray, label: np.ndarray):
        """Compute the gradient of the softmax + cross-entropy loss w.r.t. the logits.

        Recomputes softmax internally from the raw logits so that the
        returned gradient (`probabilities - label`) is correct with respect
        to `prediction`, the pre-softmax logits.

        Args:
            prediction: Raw logits (pre-softmax), shape (batch_size, n_classes).
            label: One-hot encoded targets, same shape as prediction.

        Returns:
            Gradient of the loss with respect to the logits, same shape as
            prediction.
        """

        shifted = prediction - np.max(prediction, axis=-1, keepdims=True)
        exponent = np.exp(shifted)
        probabilities = exponent / np.sum(exponent, axis=-1, keepdims=True)
        return probabilities - label

class LinearLRScheduler:
    """
    A linear learning rate scheduler that scales a neural network's

    learning rate linearly between a starting factor and an ending factor
    over a specified number of total steps.
    """

    def __init__(self, neural_network: NeuralNetwork, start_factor: float|int, end_factor: float|int, total_steps: float|int) -> None:
        """
        Initializes the linear learning rate scheduler.

        Args:
            neural_network (NeuralNetwork): The neural network instance whose
              learning rate will be managed.
            start_factor (float | int): The multiplier for the initial learning rate
              at step 0 (must be between 0 and 1).
            end_factor (float | int): The multiplier for the final learning rate at
              or after total_steps (must be between 0 and 1).
            total_steps (float | int): The total number of steps over which the
              learning rate scales down (must be > 0).
        """

        assert 0 <= start_factor <= 1, f"Start Factor Must Be Between 0 and 1 Got {start_factor}"
        assert 0 <= end_factor <= 1, f"End Factor Must Between 0 and 1 Got {end_factor}"
        assert total_steps > 0, f"Total Steps Must Be Greater Than 0 Got {total_steps}"

        self.neural_network = neural_network
        self.base_learning_rate = neural_network.learning_rate
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_steps = total_steps

        self.current_step = 0

        self.neural_network.learning_rate = self.base_learning_rate * self.start_factor

    def step(self) -> None:
        """
        Advances the scheduler by one training step, recalculates the learning

        rate using linear interpolation, and updates the neural network's learning
        rate.
        """

        self.current_step += 1
        progress = min(self.current_step / self.total_steps, 1.0)
        factor = self.start_factor + ((self.end_factor - self.start_factor) * progress)
        self.neural_network.learning_rate = self.base_learning_rate * factor

if __name__ == "__main__":
    ...