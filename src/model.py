from network import NeuralNetwork, CrossEntropyLoss, Linear, ReLU

model: NeuralNetwork = NeuralNetwork(
    layers=[
        Linear(28 * 28, 128),
        ReLU(),
        Linear(128, 64),
        ReLU(),
        Linear(64, 32),
        ReLU(),
        Linear(32, 10)
    ],
    learning_rate=0.01
)

criterion = CrossEntropyLoss

if __name__ == "__main__":
    ...
