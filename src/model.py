from package import optimizers, layers, activations
from src import schedulers, loss
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

optimizer = optimizers.StochasticGradientDescent(
    neural_network=model,
    learning_rate=0.01,
)

scheduler = schedulers.LinearLearningRateScheduler(
    optimizer=optimizer,
    start_factor=0.05,
    end_factor=1,
    transition_steps=10000,
)

if __name__ == "__main__":
    ...
