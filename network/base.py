import numpy as np

class Layer:

    def forward(self, x: np.ndarray, **kwargs) -> np.ndarray:
        ...

    def backward(self, gradients: np.ndarray, **kwargs) -> np.ndarray:
        ...

    def parameters(self) -> list:
        ...

class Loss:

    @staticmethod
    def forward(predictions: np.ndarray, labels: np.ndarray) -> float | int:
        ...

    @staticmethod
    def backward(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        ...

class Optimizer:

    def step(self, loss_gradients: np.ndarray, **kwargs) -> None:
        ...

class Scheduler:

    def step(self) -> None:
        ...

if __name__ == "__main__":
    ...