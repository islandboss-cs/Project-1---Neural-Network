import pickle
import matplotlib.pyplot as plt

##* Reading in MNIST data (provided in canvas). The data is separated into training
##* data (50 000), validation data (10 000), and test data (10 000).
with open("mnist.pkl", "rb") as f:
    data = pickle.load(f, encoding="latin1")

# Training data (50.000), validation data (10.000), test data (10.000)
(train_images, train_labels), (validation_images, validation_labels), (test_images, test_labels) = data

# Explore the data, run if main file
if __name__ == "__main__":
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