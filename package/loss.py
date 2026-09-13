from package.base import Loss, np

class MeanSquaredError(Loss):
    raise NotImplementedError

class MeanAbsoluteError(Loss):
    raise NotImplementedError

class CrossEntropyLoss(Loss):
    @staticmethod
    def forward(predictions: np.ndarray, labels: np.ndarray) -> float | int:
        shifted = predictions - np.max(predictions, axis=-1, keepdims=True)
        exponent = np.exp(shifted)
        probabilities = exponent / np.sum(exponent, axis=-1, keepdims=True)
        loss = -np.sum(labels * np.log(probabilities + 1e-9), axis=-1)
        return loss.mean()

    @staticmethod
    def backward(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        shifted = predictions - np.max(predictions, axis=-1, keepdims=True)
        exponent = np.exp(shifted)
        probabilities = exponent / np.sum(exponent, axis=-1, keepdims=True)
        return probabilities - labels