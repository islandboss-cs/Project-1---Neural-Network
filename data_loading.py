import pickle
import matplotlib.pyplot as plt
import numpy as np

##* Reading in MNIST data (provided in canvas). The data is separated into training
##* data (50 000), validation data (10 000), and test data (10 000).
with open("mnist.pkl", "rb") as f:
    data = pickle.load(f, encoding="latin1")

# Training data (50.000), validation data (10.000), test data (10.000)
(train_images, train_labels_ints), (validation_images, validation_labels_ints), (test_images, test_labels_ints) = data

def vectorize_label(label):
    # Convert integer label to vector of length 10 with 1 at label index
    vec = np.zeros(10)
    vec[label] = 1
    return vec

# Convert labels to vectors of length 10 for training
train_labels = np.array([vectorize_label(label) for label in train_labels_ints])
#validation_labels = np.array([vectorize_label(label) for label in validation_labels_ints])
validation_labels = validation_labels_ints
#test_labels = np.array([vectorize_label(label) for label in test_labels_ints])
test_labels = test_labels_ints

# Explore the data, run if main file
if __name__ == "__main__":
    print(f"Training data contains {train_images.shape[0]} elements with {train_images.shape[1]} pixels each. (28x28 pixels)")
    print(f"Training labels contain {train_labels_ints.shape[0]} elements, each one labeling its corresponding training image.")
    print()
    print(f"Validation data contains {validation_images.shape[0]} elements with {validation_images.shape[1]} pixels each. (28x28 pixels)")
    print(f"Validation labels contain {validation_labels_ints.shape[0]} elements, each one labeling its corresponding validation image.")
    print()
    print(f"Test data contains {test_images.shape[0]} elements with {test_images.shape[1]} pixels each. (28x28 pixels)")
    print(f"Test labels contain {test_labels_ints.shape[0]} elements, each one labeling its corresponding test image.")
    
    print(f"labels are: {train_labels_ints[:10]}, just integer values")
    
    print(f"training shape: {train_labels_ints.shape}\tsample shape: {train_images[0].shape}\tvalue in sample: {train_images[0, 1]}")
    print(f"type of data: {type(train_images)}, type of sample: {type(train_images[0])}")
    
    #Visualizing the first training image
    img = train_images[0]
    
    plt.imshow(img.reshape(28, 28), cmap="viridis")
    plt.title(f"Digit: {train_labels_ints[0]}")
    plt.show()