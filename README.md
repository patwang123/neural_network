This is a neural network design using the sigmoid activation function implemented using scipy and numpy.

Status: NeuralNetwork class has been fully implemented, along with all utility functions, and I am now coding the curve-sketching and main.py sections.

This was designed to be used on the MNIST handwritten digit 28x28 data, but the structure of the neural network can be adjusted for any ANN through the class methods (with some exceptions of some parameters e.g. activation function). The MNIST data is included in the data folder and can be extracted into numpy arrays through idx2numpy.

An accurate model is located in model.pickle, with the structure having a hidden layer of 28 nodes.