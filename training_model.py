####### Try loading, training, and evaluating a model
from network import FNN
from newFNN import newFNN
import numpy as np
import matplotlib.pyplot as plt
# Get the data from data_loading
from data_loading import train_images, train_labels, validation_images, validation_labels, test_images, test_labels
#print(train_images.shape[0], validation_images.shape[0])

# Creates a feedforward network with 3 layers, input (28x28), hidden (16), output(10)
network = FNN([784,16, 10])

# Training parameters
batch_size = 10
epochs = 1
learning_rate = 2.5
# Train the model
network.train_SGD(train_images, train_labels, batch_size, epochs, learning_rate, test_images, test_labels)
print(train_images[0].shape)
# Evaluate the model
print("Model after training:")
_outputs = network.evaluate(test_images, test_labels, verbose=True)


#Attack on the network by modifying the input data slightly
#Inputs the first image in test_data to the network
data_index = 1
x = test_images[data_index].reshape(784,1)
y = np.zeros((10,1))     
y[test_labels[data_index]] = 1
orig_pred = np.argmax(network.forward(x))

#Gets the gradient at the input layer
grad_input = network.input_gradient(x, y)

#Modifying the image to fool the network
epsilon = 0.05
x_atk = x + epsilon * np.sign(grad_input)
atk_pred = np.argmax(network.forward(x_atk))

# Use attack data to train the network
attack_image = [x_atk]
attack_label = [test_labels[data_index]]
#Needs more work
#network.train_SGD(attack_image, attack_label, 1, 1, 1, test_images, test_labels)

#Plot results
plt.figure(figsize=(10,6))
plt.title(f"epochs: {epochs}, batch size: {batch_size}, learning rate: {learning_rate}, epsilon: {epsilon}")
plt.subplot(1,2,1)
plt.title(f"Original\nPrediction: {orig_pred}, Correct: {test_labels[data_index]}")
plt.imshow(x.reshape(28,28), cmap='gray')
plt.axis('off')
plt.subplot(1,2,2)
plt.title(f"Attack\nPrediction: {atk_pred}, Correct: {test_labels[data_index]}")
plt.imshow(x_atk.reshape(28,28), cmap='gray')
plt.axis('off')

plt.show()