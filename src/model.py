from package import optimizers, layers, loss, activations
from package.network import NeuralNetwork

model: NeuralNetwork = NeuralNetwork(
    layers=[
        layers.Linear(784, 128),
        activations.ReLU(),
        layers.Linear(128, 128),
        activations.ReLU(),
        layers.Linear(128, 10),
    ]
)

criterion = loss.CrossEntropyLoss

optimizer = optimizers.StochasticGradientDescent(model, 0.01)

if __name__ == "__main__":
    ...
