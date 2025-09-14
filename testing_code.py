####### Test the code for loading, training a model
from main import FNN
import numpy as np
# Get the data from data_loading
from data_loading import train_images, train_labels, validation_images, validation_labels, test_images, test_labels
#print(train_images.shape[0], validation_images.shape[0])

# Create a feedforward network with 3 layers, input (28x28), hidden (16), output(10)
network = FNN([784, 16, 10])
print(network)

print("############### SIMPLE TEST ##############")
# Try running a sample through the model
output = network.forward(train_images[0].reshape(-1, 1))
print(f"Output of the model has shape: {output.shape}\nExplicitly, the output is: \n{output.reshape(1, -1)}")

# Run simple training example
training_data = train_images[:10]
training_labels = train_labels[:10]
batch_size = 5
epochs = 1
learning_rate = .01
network.train_SGD(training_data, training_labels, batch_size, epochs, learning_rate)

# Run the sample through the network again
output = network.forward(train_images[0].reshape(-1, 1))
print(f"After SGD, output of the model has shape: {output.shape}\nExplicitly, the output is: \n{output.reshape(1, -1)}")

print("################## 1000 SAMPLE TEST #############")

# More elaborate training
training_data = train_images[:1000]
training_labels = train_labels[:1000]
batch_size = 10
epochs = 5
learning_rate = .01
network.train_SGD(training_data, training_labels, batch_size, epochs, learning_rate)

# Run the sample through the network again
output = network.forward(train_images[0].reshape(-1, 1))
print(f"After SGD, output of the model has shape: {output.shape}\nExplicitly, the output is: \n{output.reshape(1, -1)}")

print("################## FULL TEST #############")

# More elaborate training
print("Model before training:")
network.evaluate(test_images, test_labels, verbose=True)

training_data = train_images
training_labels = train_labels
batch_size = 100
epochs = 10
learning_rate = .25
network.train_SGD(training_data, training_labels, batch_size, epochs, learning_rate)

# Run the sample through the network again
output = network.forward(train_images[0].reshape(-1, 1))
print(f"After SGD, output of the model has shape: {output.shape}\nExplicitly, the output is: \n{output.reshape(1, -1)}")

# Evaluate the model
print("Model after training:")
_outputs = network.evaluate(test_images, test_labels, verbose=True)