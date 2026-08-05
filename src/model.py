from network import NeuralNetwork, CrossEntropyLoss, Linear, ReLU, LinearLRScheduler

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

lr_scheduler = LinearLRScheduler(
    neural_network=model,
    start_factor=0.01,
    end_factor=1,
    total_steps=2000
)

if __name__ == "__main__":
    ...
