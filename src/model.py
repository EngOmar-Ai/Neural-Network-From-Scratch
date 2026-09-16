from network import optimizers, layers, schedulers, loss
from network import NeuralNetwork

model: NeuralNetwork = NeuralNetwork(
    layers=[
        layers.Linear(784, 128),
        layers.ReLU(),
        layers.Linear(128, 128),
        layers.ReLU(),
        layers.Linear(128, 10),
    ]
)

criterion = loss.CrossEntropyLoss

optimizer = optimizers.StochasticGradientDescent(
    neural_network=model,
    learning_rate=0.01,
)

scheduler = ...

if __name__ == "__main__":
    ...
