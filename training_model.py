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
epochs = 5
learning_rate = 3
# Train the model
network.train_SGD(train_images, train_labels, batch_size, epochs, learning_rate, test_images, test_labels)

# Evaluate the model
print("Model after training:")
_outputs = network.evaluate(test_images, test_labels, verbose=True)


##Work in progress attack
x = test_images[0].reshape(784,1)
y = np.zeros((10,1))     
y[test_labels[0]] = 1
orig_pred = np.argmax(network.forward(x))
grad_input = network.input_gradient(x, y)
epsilon = 0.3 
x_adv = x + epsilon * np.sign(grad_input)
adv_pred = np.argmax(network.forward(x_adv))


plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.title(f"Original\nPrediction: {orig_pred}, True: {test_labels[0]}")
plt.imshow(x.reshape(28,28), cmap='gray')
plt.axis('off')
plt.subplot(1,2,2)
plt.title(f"Adversarial\nPrediction: {adv_pred}")
plt.imshow(x_adv.reshape(28,28), cmap='gray')
plt.axis('off')

plt.show()
