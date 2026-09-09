from package.base import Layer, np

class Linear(Layer):

    def __init__(self, in_features: int, out_features: int):

        self.weights = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.bias = np.zeros((out_features,))

        self.input_cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:

        self.input_cache =x
        return (x @ self.weights) + self.bias

    def backward(self, gradients: np.ndarray, learning_rate: float|int) -> np.ndarray:

        batch_size = gradients.shape[0]

        dw = (self.input_cache.T @ gradients) / batch_size
        db = np.sum(gradients, axis=0) / batch_size

        next_layer_gradients = gradients @ self.weights.T

        self.weights -= dw * learning_rate
        self.bias -= db * learning_rate

        return next_layer_gradients