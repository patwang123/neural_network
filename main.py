from neuralN import NeuralNetwork
import utils
from scipy import optimize
import utils
import random
import numpy as np
import math
import idx2numpy
import matplotlib.pyplot as plt

def main():
    files = 'data/train-images.idx3-ubyte'
    arr = idx2numpy.convert_from_file(files)
    # arr is now a np.ndarray type of object of shape 60000, 28, 28
    features = ''
    labels = ''

    neural = NeuralNetwork(400,10)
    neural.set_regularization(1)
    neural.add_layer(25)
    neural.train(features,labels,100)

if __name__ == '__main__':
    main()
