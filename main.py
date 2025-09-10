import pickle
import matplotlib.pyplot as plt
import numpy as np

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
        pass

    def __repr__(self):
        return f"Contains {self.layers} layers. Input nodes: {self.in_nodes}. Output nodes: {self.out_nodes}"

# Creates a feedforward network with 3 layers, input (28x28), hidden (16), output(10)

network = FNN([784, 16, 10])
print(network)
network.create_layer(19)
print(network)
##TODO TASK 2
##* Reading in MNIST data (provided in canvas). The data is separated into training
##* data (50 000), validation data (10 000), and test data (10 000).
with open("mnist.pkl", "rb") as f:
    data = pickle.load(f, encoding="latin1")

# Training data (50.000), validation data (10.000), test data (10.000)
(train_images, train_labels), (validation_images, validation_labels), (test_images, test_labels) = data
print(f"Training data contains {train_images.shape[0]} elements with {train_images.shape[1]} pixels each. (28x28 pixels)")
print(f"Training labels contain {train_labels.shape[0]} elements, each one labeling its corresponding training image.")
print()
print(f"Validation data contains {validation_images.shape[0]} elements with {validation_images.shape[1]} pixels each. (28x28 pixels)")
print(f"Validation labels contain {validation_labels.shape[0]} elements, each one labeling its corresponding validation image.")
print()
print(f"Test data contains {test_images.shape[0]} elements with {test_images.shape[1]} pixels each. (28x28 pixels)")
print(f"Test labels contain {test_labels.shape[0]} elements, each one labeling its corresponding test image.")

#Visualizing the first training image
img = train_images[0]

plt.imshow(img.reshape(28, 28), cmap="viridis")
plt.title(f"Digit: {train_labels[0]}")
plt.show()



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
