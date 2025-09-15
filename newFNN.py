import numpy as np
from layer import Layer
from losses import f_mse, f_mse_der, f_mse2, f_mse2_der

class newFNN():
    def __init__(self, in_nodes=None, loss = "mse2"):
        if loss == "mse":
            self.loss = f_mse
            self.loss_derivative = f_mse_der
        elif loss == "mse2":
            self.loss = f_mse2
            self.loss_derivative = f_mse2_der
        else:
            print(f"Loss function {loss} not accepted.")
            return
        if in_nodes == None:
            print(f"You must initialize the FNN with a number of nodes.")
            return

        self.layers = []

    def create_layer(self, out_nodes, activation="sigmoid"):
        if not self.layers:
            in_nodes = self.input_size
        else:
            in_nodes = self.layers[-1].W.shape[0]  

        self.layers.append(Layer(in_nodes, out_nodes, activation))

    def forward_propagation(self, data):
        a = data
        values = [data]

        for layer in self.layers:
            a = layer.forward(a)
            values.append(a)
        return a, values
    
    def backward_propagation(self, data, real_data):
        prediction_data, activations = self.forward_propagation()
        delta = self.loss_derivative(real_data, prediction_data)
        grads_w = []
        grads_b = []

        for i in reversed(range(len(self.layers))):
            present_layer = self.layers[i]
            previous_a = activations[i]

            dw, db, delta = present_layer.backward(delta, previous_a)
            # Note: With insert, gradient list can be builded from the end to start
            grads_w.insert(0, dw)
            grads_b.insert(0, db)

        return grads_w, grads_b
    
    # Taken from the last update, thanks
    def train_SGD(self, training_data, training_labels, batch_size, epochs, learning_rate = 0.01):
        # Take training data, mini-batch size, and number of epochs to train over
        # Epochs are complete passes over the data.
        # Also use a learning rate to scale the gradient shift
        # Randomize training data and labels, keeping indices matching
        # There should be roughly size(training_data)/mini-batch size batches per epoch
        # Number of whole batches, there likely be some left over for a final partial batch
        whole_batch_num = training_data.shape[0] // batch_size

        # Repeat for each epoch
        for epoch in range(epochs):
            rand_inds = np.arange(training_data.shape[0])
            np.random.shuffle(rand_inds)
            training_data = training_data[rand_inds]
            training_labels = training_labels[rand_inds]

            epoch_loss = 0
            # For each batch, there will be N whole batches and a final partial batch
            for b in range(whole_batch_num + 1):
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
                # To avoid an empty batch
                if training_subset.shape[0] == 0:
                    continue

                # Forward propagation
                prediction, _ = self.forward_propagation(training_subset)
                batch_loss = self.loss(label_subset, prediction)

                epoch_loss += batch_loss

                # Backward propagation
                grads_w, grads_b = self.backward_propagation(training_subset, label_subset)

                for layer, dw, db in zip(self.layers, grads_w, grads_b):
                    layer.w -= learning_rate * dw
                    layer.b -= learning_rate * db

                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")





        