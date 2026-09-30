from src.data import load_testing_data, load_training_data
from src.model import model, criterion, optimizer

import numpy as np

def train(epochs: int):

    model.train()

    for epoch in range(epochs):

        training_counter = 0
        training_loss = 0

        for x, y in load_training_data(batch_size=32):

            prediction = model.forward(x)

            loss = criterion.forward(prediction, y)

            error = criterion.backward(prediction, y)
            model.backward(error)

            optimizer.step()
            optimizer.zero_grad()

            training_counter += 1
            training_loss += loss

        validation = validate()
        print(f"Epoch {epoch + 1}/{epochs}: Training Loss = {training_loss / training_counter} | Validation Loss = {validation}")

def validate():

    model.eval()

    validation_counter = 0
    validation_loss = 0

    for x, y in load_testing_data(batch_size=32):

        prediction = model.forward(x)

        loss = criterion.forward(prediction, y)

        validation_counter += 1
        validation_loss += loss

    model.train()

    return validation_loss / validation_counter

def test():

    model.eval()

    predictions = []
    labels = []

    for x, y in load_testing_data(batch_size=32):

        prediction = np.argmax(model.forward(x), axis=1).tolist()
        label = np.argmax(y, axis=1).tolist()

        predictions.append(prediction)
        labels.append(label)

    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)

    accuracy = np.mean(predictions == labels)

    print(f"Test Accuracy = {accuracy * 100}%")


if __name__ == "__main__":
    ...