import utils
import idx2numpy
from neuralN import NeuralNetwork
import time

files = {'train_images': 'data/train-images.idx3-ubyte',
         'train_labels': 'data/train-labels.idx1-ubyte',
         'test_images': 'data/t10k-images.idx3-ubyte',
         'test_labels': 'data/t10k-labels.idx1-ubyte'}
files = {k: utils.format_training(idx2numpy.convert_from_file(v)) for k, v in files.items()}

f = lambda x: files[x]

size = len(f('train_labels'))

#default parameters
m_ = 0.5 #takes too long to train the entire thing
LEARNING_RATE = 1.0
REGULARIZATION = 1.0
NUM_INPUTS = 28*28
NUM_LABELS = 10
HIDDEN_LAYERS = [28]
MAX_ITERATIONS = 100


def initialize_nn():
    neural = NeuralNetwork(NUM_INPUTS,
                           NUM_LABELS)
    neural.set_regularization(REGULARIZATION)
    neural.set_learning_rate(LEARNING_RATE)
    for layer in HIDDEN_LAYERS:
        neural.add_layer(layer)
    return neural
"""
constructs learning curves based on the # of training items to be used
"""
def construct_learning_curves(percentages):
    errors = {}
    train_images = f('train_images')[:int(m_*size)]
    train_labels = f('train_labels')[:int(m_*size)]
    for p in percentages:
        nn = initialize_nn()
        print('Training... {0} percent of the training data'.format(p))
        t1 = time.time()
        nn.train(train_images,
                    train_labels,
                    MAX_ITERATIONS)
        s = time.time()-t1
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        print('Time elapsed: {0} hrs, {1} mins, {2} secs'.format(h, m, s))
        
        print('Testing training data...')
        train_error = nn.test(train_images,
                                train_labels)

        print('Testing testing data...')
        test_error = nn.test(f('test_images'),
                                f('test_labels'))
        errors[p] = [train_error,test_error]
    return errors


def construct_learning_curves(percentages):
    errors = {}
    train_images = f('train_images')[:int(m_*size)]
    train_labels = f('train_labels')[:int(m_*size)]
    for p in percentages:
        nn = initialize_nn()
        print('Training... {0} percent of the training data'.format(p))
        t1 = time.time()
        nn.train(train_images,
                 train_labels,
                 MAX_ITERATIONS)
        s = time.time()-t1
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        print('Time elapsed: {0} hrs, {1} mins, {2} secs'.format(h, m, s))

        print('Testing training data...')
        train_error = nn.test(train_images,
                              train_labels)

        print('Testing testing data...')
        test_error = nn.test(f('test_images'),
                             f('test_labels'))
        errors[p] = [train_error, test_error]
    return errors
