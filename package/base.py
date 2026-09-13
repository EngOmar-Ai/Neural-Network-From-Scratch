import numpy as np

class Layer:
    """Abstract base class for trainable layers (e.g. Linear).

    Subclasses are expected to hold their own parameters and implement both
    `forward` and `backward`, with `backward` responsible for computing
    parameter gradients, updating parameters in place, and returning the
    gradient to propagate to the previous layer.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Computes the layer's output for a given input.

        Args:
            x: Input array, typically of shape (batch_size, in_features).

        Returns:
            Output array produced by this layer.
        """
        ...

    def backward(self, gradients: np.ndarray, learning_rate: float | int) -> np.ndarray:
        """Computes gradients, updates parameters in place, and backpropagates.

        Args:
            gradients: Gradient of the loss with respect to this layer's
                output.
            learning_rate: Step size used to update this layer's parameters.

        Returns:
            Gradient of the loss with respect to this layer's input, to be
            passed to the previous layer.
        """
        ...


class Activation:
    """Abstract base class for parameter-free activation functions (e.g. ReLU).

    Subclasses implement an elementwise (or otherwise parameter-free)
    transformation in `forward`, and compute the gradient with respect to
    the input in `backward`. Unlike `Layer.backward`, `Activation.backward`
    takes no learning rate since there are no parameters to update.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Applies the activation function to the input.

        Args:
            x: Input array of any shape.

        Returns:
            Output array, same shape as x.
        """
        ...

    def backward(self, gradients: np.ndarray) -> np.ndarray:
        """Computes the gradient of the loss with respect to the input.

        Args:
            gradients: Gradient of the loss with respect to this
                activation's output, same shape as the cached input.

        Returns:
            Gradient of the loss with respect to this activation's input,
            same shape as gradients.
        """
        ...


class Optimizer:
    """Abstract base class for optimizers (e.g. StochasticGradientDescent).

    Subclasses drive backpropagation through a network's layers, using
    `loss_gradients` as the starting point and updating each layer's
    parameters as gradients are propagated backward.
    """

    def step(self, loss_gradients: np.ndarray) -> None:
        """Performs one optimization step, updating the network's parameters.

        Args:
            loss_gradients: Gradient of the loss with respect to the
                network's final output.

        Returns:
            None. Parameters are updated in place.
        """
        ...


class Loss:
    """Abstract base class for loss functions.

    Subclasses implement `forward` to compute a scalar loss value from
    predictions and labels, and `backward` to compute the gradient of that
    loss with respect to the predictions (the starting point for
    backpropagation). Both methods are static since loss functions are
    typically stateless.
    """

    @staticmethod
    def forward(predictions: np.ndarray, labels: np.ndarray) -> float | int:
        """Computes the scalar loss value.

        Args:
            predictions: Model predictions, typically of shape
                (batch_size, ...).
            labels: Ground-truth labels, same shape as predictions.

        Returns:
            Scalar loss value.
        """
        ...

    @staticmethod
    def backward(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Computes the gradient of the loss with respect to predictions.

        Args:
            predictions: Model predictions, same shape as passed to
                `forward`.
            labels: Ground-truth labels, same shape as predictions.

        Returns:
            Gradient array, same shape as predictions.
        """
        ...


class Scheduler:
    """Abstract base class for learning-rate (or other hyperparameter) schedulers.

    Subclasses implement `step` to advance the schedule by one unit
    (typically one epoch or one training step), usually mutating some
    associated optimizer's `learning_rate` in place.
    """

    def step(self) -> None:
        """Advances the schedule by one step.

        Returns:
            None. Typically, mutates an associated optimizer's state
            (e.g. `learning_rate`) in place.
        """
        ...