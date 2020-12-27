This is a neural network design using the sigmoid activation function implemented using scipy and numpy.

Status: Done! I'll add new features if I think of any.

This was designed to be used on the MNIST handwritten digit 28x28 data, but the structure of the neural network can be adjusted for any ANN through the class methods (with some exceptions of some parameters e.g. activation function). The MNIST data is included in the data folder and can be extracted into numpy arrays through idx2numpy.

An accurate model is located in model.pickle, with the structure having a hidden layer of 28 nodes.

For NN diagnosis based on the regularization parameter lambda or the learning curve (# of training examples), run create_curve.py. This will run for each of the parameters, and will upload it to a pickle file that can be loaded in for use. This will take a while to run, since many NNs are being trained.