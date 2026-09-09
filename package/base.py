import numpy as np

class Layer:

    def forward(self, x: np.ndarray) -> np.ndarray:
        ...
    def backward(self, gradients: np.ndarray, learning_rate: float|int) -> np.ndarray:
        ...

class Activation:

    def forward(self, x: np.ndarray) -> np.ndarray:
        ...

    def backward(self, gradients: np.ndarray) -> np.ndarray:
        ...

class LossFunction:

    @staticmethod
    def forward(prediction: np.ndarray, label: np.ndarray) -> float | int:
        ...

    @staticmethod
    def backward(prediction: np.ndarray, label: np.ndarray) -> np.ndarray:
        ...

class Optimizer:
    def step(self, loss_gradients: np.ndarray) -> None:
        ...

