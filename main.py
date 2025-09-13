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
        self.layers.append(nodes_layer)

    def forward(self, data):
        self.a = [data]
        self.z = []

        # Haven't tried it but it's supose to work
        for w, b in zip(self.w, self.b):
            z = self.a[-1] @ w + b
            a = f_sigmoid(z)

            self.z.append(z)
            self.a.append(a)

    """def backward(self, true_result):
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
        
        return grads_w, grads_b"""
    
    def back_propagation(self, true_result):
        """
        Note: Assumes self.forward has been run on the corresponding sample
        
        Parameters
        ----------
        true_result : ndarray
            The 10 element array representing the label of the sample

        Returns
        -------
        grads_w : list
            List of 2d numpy arrays containing the weight gradients for this sample
        grads_b : list
            List of numpy arrays containing the bias gradients for this sample

        """
        # We have activations self.a and z values self.z internal to the class
        # Initialize lists of gradients of same dimensions as weights and biases
        grads_w = [np.zeros(layer.shape) for layer in self.w]
        grads_b = [np.zeros(layer.shape) for layer in self.b]
        # delta for final layer, to be recursively multiplied
        # delta final layer is the componentwise product of the derivative of the loss
        # function w.r.t the activations and the derivative of the activation function 
        # of the last layer
        delta = (self.a[-1] - true_result) * f_sigmoid_der(self.z[-1])
        # Set final grads from delta
        grads_w[-1] = np.dot(delta, self.a[-2].T)
        grads_b[-1] = delta
        # Repeat all the way to the first layer
        # Note that len(self.z)=len(self.b)=len(self.w)=self.n_layers-1, but len(self.a)==self.n_layers
        # So we start at index n_layers-3 since we already computed the values for the last layer
        # and the last index in self.w etc... is n_layers-2
        for l in range(self.n_layers-3, -1, -1): # range: n_layers-3, n_layers-1,..., 0
            # Since the lengths of the arrays are different, the index l corresponds to the
            # index in the weights array. The corresponding index in self.a is l+1
            # So we need to add 1 to the index whenever referring to the corresponding set
            # of activations self.a[l+1] that go with self.w[l], self.z[l], self.b[l]
            
            # New delta is the matrix multiplication of the transpose of weight matrix
            # at the deeper level with the previous delta, multiplied componentwise
            # with the derivative of the activation function w.r.t the z vector
            d_sig_dz = f_sigmoid_der(self.z[l])
            delta = np.dot(self.w[l+1].T, delta) * d_sig_dz
            # Set this next set of gradients at this layer
            grads_w[l] = np.dot(delta, self.a[l].T) # a[l] not a[l-1] because a offset by inputs in element 0
            grads_b[l] = delta
        # A gradient has been computed for every weight and bias
        return grads_w, grads_b

    def train_SGD(self, training_data, training_labels, batch_size, epochs, learning_rate):
        # Take training data, mini-batch size, and number of epochs to train over
        # Epochs are complete passes over the data.
        # Also use a learning rate to scale the gradient shift
        
        # Randomize training data and labels, keeping indices matching
        rand_inds = np.random.permutation(len(training_data))
        training_data = training_data[rand_inds]
        training_labels = training_labels[rand_inds]
        
        # There should be roughly size(training_data)/mini-batch size batches per epoch
        # Number of whole batches, there likely be some left over for a final partial batch
        whole_batch_num = training_data.shape[0] // batch_size
        # Repeat for each epoch
        for e in range(epochs):
            # For each batch, there will be N whole batches and a final partial batch
            for b in range(whole_batch_num + 1):
                # Gradient sum for weights and biases, initialize to zeros
                weight_gradient_sum = [np.zeros(layer.shape) for layer in self.w]
                bias_gradient_sum = [np.zeros(layer.shape) for layer in self.b]
                # Batches of data and labels
                # If we aren't on that final partial batch, create a full batch
                # of the desired size
                if b != whole_batch_num:
                    training_subset = training_data[b*batch_size:(b+1)*batch_size]
                    label_subset = training_labels[b*batch_size:(b+1)*batch_size]
                # Else, we are on the final partial batch, so just take the remaining samples
                else:
                    training_subset = training_data[b*batch_size:]
                    label_subset = training_labels[b*batch_size:]
                # Take each sample, run it forward through the network, get the output
                # Then run it backward and get the gradients
                # Add these gradients to a running sum
                # Mostly there will be full batches, but with one final one, so use the
                # number of samples to iterate through them
                current_batch_size = training_subset.shape[0]
                for i in range(current_batch_size):
                    sample = training_subset[i]
                    label = label_subset[i]
                    # Run forward
                    self.forward(sample)
                    # Run backward and get the gradients
                    grads_w, grads_b = self.backward(label)
                    # Update running sums, adding numpy arrays at each index componentwise
                    for j in range(len(grads_w)):
                        weight_gradient_sum[j] += grads_w[j]
                        bias_gradient_sum[j] += grads_b[j]
                # Gradient averages over the batch are done, update the weights and biases
                for k in range(len(self.w)):
                    self.w[k] += -1*learning_rate*weight_gradient_sum[k] / current_batch_size
                    self.b[k] += -1*learning_rate*bias_gradient_sum[k] / current_batch_size
            

    def __repr__(self):
        return f"Contains {len(self.layers)} layers. Input nodes: {self.in_nodes}. Output nodes: {self.out_nodes}. All nodes: {self.layers}"

# Creates a feedforward network with 3 layers, input (28x28), hidden (16), output(10)


##! TESTING THE FNN CLASS
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
