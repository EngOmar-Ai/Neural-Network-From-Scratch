from typing import Generator
import csv

import numpy as np

TRAIN_FILE_PATH = '../data/mnist_train.csv'
TEST_FILE_PATH = '../data/mnist_test.csv'

def load_training_data(batch_size: int) -> Generator:
    """
    Stream the MNIST training set from CSV in mini-batches.

    Reads TRAIN_FILE_PATH one row at a time (skipping the header), where
    each row is expected to be a class label followed by 784 pixel values.
    Pixel values are normalized to [0, 1] and labels are one-hot encoded.
    Rows are accumulated into batches and yielded once a batch is full; any
    trailing rows that don't fill a complete batch are dropped.

    Args:
        batch_size: Number of samples per yielded batch.

    Yields:
        Tuples of (x, y) where:
            x: Input batch, shape (batch_size, 784), pixel values in [0, 1].
            y: One-hot encoded labels, shape (batch_size, 10).
    """

    with open(TRAIN_FILE_PATH, 'r') as file:

        reader = csv.reader(file)
        next(reader)

        batch_input = []
        batch_output = []

        for row in reader:

            input_data = list(map(float, row[1:]))
            output_data = int(row[0])

            x = np.array(input_data) / 255.0
            y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

            y[output_data] = 1

            batch_input.append(x)
            batch_output.append(y)

            if len(batch_input) == batch_size:
                yield np.stack(batch_input), np.stack(batch_output)
                batch_input = []
                batch_output = []

def load_testing_data(batch_size: int) -> Generator:
    """
    Stream the MNIST test set from CSV in mini-batches.

    Reads TEST_FILE_PATH one row at a time (skipping the header), where
    each row is expected to be a class label followed by 784 pixel values.
    Pixel values are normalized to [0, 1] and labels are one-hot encoded.
    Rows are accumulated into batches and yielded once a batch is full; any
    trailing rows that don't fill a complete batch are dropped.

    Args:
       batch_size: Number of samples per yielded batch.

    Yields:
       Tuples of (x, y) where:
           x: Input batch, shape (batch_size, 784), pixel values in [0, 1].
           y: One-hot encoded labels, shape (batch_size, 10).
    """

    with open(TEST_FILE_PATH, 'r') as file:

        reader = csv.reader(file)
        next(reader)

        batch_input = []
        batch_output = []

        for row in reader:

            input_data = list(map(float, row[1:]))
            output_data = int(row[0])

            x = np.array(input_data) / 255.0
            y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

            y[output_data] = 1

            batch_input.append(x)
            batch_output.append(y)

            if len(batch_input) == batch_size:
                yield np.stack(batch_input), np.stack(batch_output)
                batch_input = []
                batch_output = []

if __name__ == '__main__':
    ...




