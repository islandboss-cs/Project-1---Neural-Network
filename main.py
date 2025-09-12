import matplotlib.pyplot as plt
import numpy as np

# Get the data from data_loading
from data_loading import train_images, train_labels, validation_images, validation_labels, test_images, test_labels
print(train_images.shape[0], validation_images.shape[0])

##! PROJECT 1 - Neural Network

##TODO TASK 1
##* Implement a feedforward neural network (as a class) consisting of 3 layers (input,
##* hidden, output layer), where each layer can contain any number of neurons. Use
##* the sigmoid function as the activation function.

# EXTRA: Possible new FN0N
# sigmoid and its derivative functions are implemented otside the class
def f_sigmoid(x):
    return 1 / (1 + np.exp(-x))

def f_sigmoid_der(x):
    return f_sigmoid(x)*(1 - f_sigmoid(x))


class FNN():
    # This FNN creates layers that will always use sigmoid function as defeault activation function
    # ga

    def __init__(self, nodes):
        """
        nodes (array): number of nodes for each layer
        """

        self.layers = list(nodes)
        print(self.layers)
        self.n_layers = len(nodes) 
        print(self.n_layers)
        
        # Creating matrixes for W and B
        self.w = []
        self.b = []

        self.in_nodes = nodes[0]
        self.out_nodes = nodes[-1]


    def create_layer(self, nodes_layer):
        w_layer = np.random.randn(self.out_nodes, nodes_layer)
        b_layer = np.zeros((nodes_layer))

        self.w.append(w_layer)
        self.b.append(b_layer)
        
        self.out_nodes = nodes_layer

    def forward(self, data):
        self.a = [data]
        self.z = []

        # Haven't tried it but it's supose to work
        for w, b in zip(self.w, self.b):
            z = self.a[-1] @ w + b
            a = f_sigmoid(z)

            self.z.append(z)
            self.a.append[a]

    def backward(self, true_result):
        grads_w = []
        grads_b = []

        error = self.a[-1] - true_result
        delta = error * f_sigmoid_der(self.a[-1])

        for i in reversed(range(len(self.w))):
            dw = self.a[i].T @ delta / true_result.shape[0]
            db = np.mean(delta, axis=0, keepdims=True)
            grads_w.insert(0, dw)
            grads_b.insert(0, db)

            if i > 0:
                delta = (delta @ self.w[i].T) * (self.a[i] * (1 - self.a[i]))

    def train_SGD():
		# Take training data, mini-batch size, and number of epochs to train over
		# Epochs are complete passes over the data.
		# There should be roughly size(training_data)/mini-batch size batches per epoch
		
		
		
        pass

    def __repr__(self):
        return f"Contains {self.layers} layers. Input nodes: {self.in_nodes}. Output nodes: {self.out_nodes}"

# Creates a feedforward network with 3 layers, input (28x28), hidden (16), output(10)

network = FNN([784, 16, 10])
print(network)
network.create_layer(19)
print(network)
##TODO TASK 2



##TODO TASK 3
##* Implement the stochastic gradient method (SGD) to train the network. The im-
##* plementation of the SGD should allow for different mini-batch sizes and different
##* numbers of epochs. An epoch is the complete pass of the training data through
##* the learning algorithm.




##TODO TASK 4
##* Implement the backpropagation algorithm (used in SGD to effectively calculate
##* the derivative).




##TODO TASK 5
##* Train and test the accuracy of the network for the following parameters:
##* • Input layer with 784 + 1 neurons
##* • hidden layer with 30 + 1 neurons
##* • Output layer with 10 neurons
##* As loss function use the quadratic function (square loss)
##* 1/2n ∑_x ||h_w(x) - y||_2 ^2
##* where (x, y) is a pair of training data, n the amount of used training data, and hw
##* represents the neural network.




##TODO TASK 6
##* Print an output of the learning success per epoch.




##TODO TASK 7
##* Implement an attack on the trained neural network.
