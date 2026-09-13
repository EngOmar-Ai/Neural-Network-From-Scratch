from src.data import load_testing_data, load_training_data
from src.model import model, criterion, optimizer, scheduler

import numpy as np

def train(epochs: int):
    """
    Trains the neural network model for a given number of epochs.

    Iterates over the training data in batches, performs forward and backward passes,
    updates model weights, and evaluates validation loss at the end of each epoch.

    Args:
        epochs (int): The number of complete passes through the training dataset.
    """

    for epoch in range(epochs):

        training_counter = 0
        training_loss = 0

        for x, y in load_training_data(batch_size=32):

            prediction = model.forward(x)

            loss = criterion.forward(prediction, y)

            error = criterion.backward(prediction, y)

            optimizer.step(error)
            scheduler.step()

            training_counter += 1
            training_loss += loss

        validation = validate()
        print(f"Epoch {epoch + 1}/{epochs}: Training Loss = {training_loss / training_counter} | Validation Loss = {validation}")

def validate():
    """
    Evaluates the model on the testing/validation dataset.

    Computes the average loss across all batches in the validation set without
    performing backpropagation.

    Returns:
        float: The average validation loss across the dataset.
    """

    validation_counter = 0
    validation_loss = 0

    for x, y in load_testing_data(batch_size=32):
        prediction = model.forward(x)
        loss = criterion.forward(prediction, y)

        validation_counter += 1
        validation_loss += loss

    return validation_loss / validation_counter

def test():
    """
    Tests the model performance and calculates classification accuracy.

    Runs inference on the test dataset, extracts predicted and true classes using
    argmax, and prints the overall percentage accuracy.

    """

    predictions = []
    labels = []

    for x, y in load_testing_data(batch_size=32):

        prediction = np.argmax(model.forward(x), axis=1).tolist()
        label = np.argmax(y, axis=1).tolist()

        predictions.append(prediction)
        labels.append(label)

    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)

    accuracy = (predictions == labels).mean()

    print(f"Test Accuracy: {accuracy * 100}%")

if __name__ == "__main__":
    ...