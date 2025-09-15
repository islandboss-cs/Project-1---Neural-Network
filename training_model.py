####### Try loading, training, and evaluating a model
from network import FNN
import numpy as np
from drawing import DigitDrawApp
# Get the data from data_loading
from data_loading import train_images, train_labels, validation_images, validation_labels, test_images, test_labels
#print(train_images.shape[0], validation_images.shape[0])

# Creates a feedforward network with 3 layers, input (28x28), hidden (16), output(10)
network = FNN([784, 16, 10])

# Training parameters
batch_size = 10
epochs = 30
learning_rate = 3
# Train the model
network.train_SGD(train_images, train_labels, batch_size, epochs, learning_rate, test_images, test_labels)

# Evaluate the model
print("Model after training:")
_outputs = network.evaluate(test_images, test_labels, verbose=True)

