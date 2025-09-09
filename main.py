import pickle
import matplotlib.pyplot as plt
import numpy as np

##! PROJECT 1 - Neural Network
##? Placeholder task assignments

##TODO TASK 1
##? Main responsibility: Rafael
##* Implement a feedforward neural network (as a class) consisting of 3 layers (input,
##* hidden, output layer), where each layer can contain any number of neurons. Use
##* the sigmoid function as the activation function.
class FFN():

    def __init__(self, input: int, hidden: int, output: int):
        '''
        Three layers, input hidden output
        '''

        self.input = input
        self.hidden = hidden
        self.output = output

    
    def sigmoid():
        pass



##TODO TASK 2
##? Main responsibility: Edvin
##* Reading in MNIST data (provided in canvas). The data is separated into training
##* data (50 000), validation data (10 000), and test data (10 000).
with open("mnist.pkl", "rb") as f:
    data = pickle.load(f, encoding="latin1")

# Training data (50.000), validation data (10.000), test data (10.000)
(train_images, train_labels), (validation_images, validation_labels), (test_images, test_labels) = data

img = train_images[0]

plt.imshow(img.reshape(28, 28), cmap="gray")
plt.title(f"Label: {train_labels[0]}")
plt.show()



##TODO TASK 3
##? Main responsibility: Las
##* Implement the stochastic gradient method (SGD) to train the network. The im-
##* plementation of the SGD should allow for different mini-batch sizes and different
##* numbers of epochs. An epoch is the complete pass of the training data through
##* the learning algorithm.




##TODO TASK 4
##? Main responsibility: Martin
##* Implement the backpropagation algorithm (used in SGD to effectively calculate
##* the derivative).




##TODO TASK 5
##? Main responsibility: Rafael
##* Train and test the accuracy of the network for the following parameters:
##* • Input layer with 784 + 1 neurons
##* • hidden layer with 30 + 1 neurons
##* • Output layer with 10 neurons
##* As loss function use the quadratic function (square loss)
##* 1/2n ∑_x ||h_w(x) - y||_2 ^2
##* where (x, y) is a pair of training data, n the amount of used training data, and hw
##* represents the neural network.




##TODO TASK 6
##? Main responsibility: Las
##* Print an output of the learning success per epoch.




##TODO TASK 7
##? Main responsibility: Edvin
##* Implement an attack on the trained neural network.
