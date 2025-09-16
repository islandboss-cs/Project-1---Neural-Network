import numpy as np
import matplotlib.pyplot as plt

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
        # Initialize the model with input layer
        # This doesn't have a weight matrix or bias vector
        self.layers = [nodes[0]]
        #print(self.layers)
        self.n_layers = 1
        #print(self.n_layers)
        self.accuracy = []
        # Creating matrixes for W and B
        self.w = []
        self.b = []
        
        self.in_nodes = self.out_nodes = nodes[0]
        
        # Create the remaining layers, each of these does have a weight matrix and bias vector
        for num_nodes in nodes[1:]:
            self.create_layer(num_nodes)


    def create_layer(self, nodes_layer):
        # Create a layer with weight matrix of size nodes_layer x self.out_nodes
        # This way, a step is computed as w x previous activations + b
        w_layer = np.random.randn(nodes_layer, self.out_nodes)
        # Create nodes_layer biases, one for each node
        b_layer = np.ones((nodes_layer, 1))

        self.w.append(w_layer)
        self.b.append(b_layer)
        
        self.out_nodes = nodes_layer
        self.layers.append(nodes_layer)
        self.n_layers += 1

    def forward(self, data):
        self.a = [data]
        self.z = []

        # Take the input data vector and multiply it through each layer, one after the other
        for w, b in zip(self.w, self.b):
            # previous activations x weights + biases
            z = np.matmul(w, self.a[-1]) + b
            a = f_sigmoid(z)

            self.z.append(z)
            self.a.append(a)
        
        # Return output of last layer
        return self.a[-1]

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
        #print(f"a shape: {self.a[-1].shape}, true_result shape: {true_result.shape}, sig_der shape: {f_sigmoid_der(self.z[-1]).shape}")
        delta = (self.a[-1] - true_result) * f_sigmoid_der(self.z[-1])
        #print(f"delta shape after: {delta.shape}")
        #print(f"Previous activation transpose shape: {self.a[-2].T.shape}")
        # Set final grads from delta
        grads_w[-1] = np.matmul(delta, self.a[-2].T)
        grads_b[-1] = delta
        #print(f"After grads calc: weight shape: {grads_w[-1].shape}, bias shape: {grads_b[-1].shape}")
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
            delta = np.matmul(self.w[l+1].T, delta) * d_sig_dz
            # Set this next set of gradients at this layer
            grads_w[l] = np.matmul(delta, self.a[l].T) # a[l] not a[l-1] because a offset by inputs in element 0
            grads_b[l] = delta
        # A gradient has been computed for every weight and bias
        return grads_w, grads_b

    def train_SGD(self, training_data, training_labels, batch_size, epochs, learning_rate, test_data=None, test_labels=None):
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
            if test_data is not None:
                self.evaluate(test_data, test_labels, True)
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
                # Only do this if there are at least enough for half a batch
                else:
                    training_subset = training_data[b*batch_size:]
                    label_subset = training_labels[b*batch_size:]
                    if len(training_subset) < batch_size / 2:
                        continue
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
                    _output = self.forward(sample.reshape(-1, 1))
                    # Run backward and get the gradients
                    grads_w, grads_b = self.back_propagation(label.reshape(-1, 1))
                    # Update running sums, adding numpy arrays at each index componentwise
                    for j in range(len(grads_w)):
                        weight_gradient_sum[j] += grads_w[j]
                        bias_gradient_sum[j] += grads_b[j]
                # Gradient averages over the batch are done, update the weights and biases
                for k in range(len(self.w)):
                    self.w[k] += -1*learning_rate*weight_gradient_sum[k] / current_batch_size
                    self.b[k] += -1*learning_rate*bias_gradient_sum[k] / current_batch_size
# x matches the length of accuracy

        self.plot_accuracy(epochs, batch_size, learning_rate)
        
    def evaluate(self, test_data, test_labels, verbose=False):
        # Run the model on each sample in the test data and compare the output to the correct one
        # Assume test_labels are ints of the correct output for the sample
        # Return an array of the model outputs on the testing data
        outputs = np.array([self.forward(sample.reshape(-1, 1)) for sample in test_data])
        int_outputs = np.array([self.vector_to_label(vec) for vec in outputs])
        if verbose:
            #print(int_outputs)
            #print(test_labels)
            correct_bools = int_outputs == test_labels
            total_correct = np.sum(correct_bools)

            self.accuracy.append(total_correct*100/len(correct_bools))
            print(f"Performance on test data: {total_correct}/{len(correct_bools)}, {total_correct*100/len(correct_bools):.2f}% acc")
        return outputs
    
    #Gradient at the input, maybe works ?
    def input_gradient(self, x, y):
        self.forward(x)

        delta = (self.a[-1] - y) * f_sigmoid_der(self.z[-1])
        for l in range(self.n_layers-3, -1, -1):
            delta = np.matmul(self.w[l+1].T, delta) * f_sigmoid_der(self.z[l])
        return np.matmul(self.w[0].T, delta) * f_sigmoid_der(self.a[0])
        
    def vector_to_label(self, vec):
        return int(np.argmax(vec))
    
    def plot_accuracy(self, epochs, batch_size, learning_rate):
        #x: epochs
        #y: accuracy of predictions per epoch
        x = np.arange(1, len(self.accuracy) + 1)
        plt.plot(x, self.accuracy, marker="o")

        plt.title(f"epochs: {epochs}, batch size: {batch_size}, learning rate: {learning_rate}")
        
        #Plot y between 0 and 100%
        plt.ylim(0, 101)
        plt.yticks(range(0, 101, 10))
        
        #x ticks at each epoch
        plt.xticks(x)

        # grid
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        
        plt.show()
    
    def __repr__(self):
        return f"Contains {len(self.layers)} layers. Input nodes: {self.in_nodes}. Output nodes: {self.out_nodes}. All nodes: {self.layers}"
