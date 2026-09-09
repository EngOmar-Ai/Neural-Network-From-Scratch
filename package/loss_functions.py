from package.base import LossFunction, np

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