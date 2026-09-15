import numpy as np

class Layer:

    def forward(self, x: np.ndarray, **kwargs) -> np.ndarray:
        ...

    def backward(self, gradients: np.ndarray, **kwargs) -> np.ndarray:
        ...

class Optimizer:

    def step(self, loss_gradients: np.ndarray, **kwargs) -> None:
        ...

class Scheduler:

    def step(self) -> None:
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

if __name__ == "__main__":
    ...