from network import NeuralNetwork, optimizers, layers, loss

model: NeuralNetwork = NeuralNetwork(
    layers=[
        layers.Linear(28 * 28, 128),
        layers.ReLU(),
        layers.Dropout(0.2),
        layers.Linear(128, 64),
        layers.ReLU(),
        layers.Dropout(0.2),
        layers.Linear(64, 10),
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
