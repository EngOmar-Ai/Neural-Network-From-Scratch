from network.base import Loss, np

class MeanSquaredError(Loss):
    ...

class MeanAbsoluteError(Loss):
    ...

class CrossEntropyLoss(Loss):
    """
    Softmax cross-entropy loss for multi-class classification.

    Combines a numerically-stable softmax with a cross-entropy loss.
    `predictions` are expected to be raw, unnormalized logits (softmax is
    applied internally), and `labels` are expected to be one-hot (or
    otherwise valid probability distributions) over the last axis.
    """

    @staticmethod
    def forward(predictions: np.ndarray, labels: np.ndarray) -> float | int:
        """
        Computes the mean softmax cross-entropy loss over the batch.

        Applies a numerically-stable softmax to `predictions` (subtracting
        the max before exponentiating), then computes cross-entropy against
        `labels`. A small epsilon (1e-9) is added inside the log to avoid
        taking log(0).

        Args:
            predictions: Raw logits of shape (batch_size, num_classes).
            labels: One-hot (or soft) target distribution, same shape as
                predictions.

        Returns:
            Scalar loss value, averaged over the batch.
        """

        shifted = predictions - np.max(predictions, axis=-1, keepdims=True)
        exponent = np.exp(shifted)
        probabilities = exponent / np.sum(exponent, axis=-1, keepdims=True)
        loss = -np.sum(labels * np.log(probabilities + 1e-9), axis=-1)
        return loss.mean()

    @staticmethod
    def backward(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the loss with respect to the logits.

        Recomputes the softmax probabilities and returns `probabilities -
        labels`, which is the simplified gradient of softmax + cross-entropy
        combined with respect to the raw logits.

        Args:
            predictions: Raw logits, same shape as passed to `forward`.
            labels: One-hot (or soft) target distribution, same shape as
                predictions.

        Returns:
            Gradient array, same shape as predictions. Note this is not yet
            divided by batch size — that division happens in `Layer.backward`
            (e.g. in `Linear`), so make sure whatever consumes this gradient
            handles batch averaging consistently.
        """

        shifted = predictions - np.max(predictions, axis=-1, keepdims=True)
        exponent = np.exp(shifted)
        probabilities = exponent / np.sum(exponent, axis=-1, keepdims=True)
        return probabilities - labels