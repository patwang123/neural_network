from neuralN import NeuralNetwork
import utils
import idx2numpy
import matplotlib.pyplot as plt
import time
import pickle

m_ = 0.001  # fraction of training dataset for learning curve
LEARNING_RATE = 1.0 #1.0 is standard.
REGULARIZATION = 1.0
NUM_INPUTS = 28*28
NUM_LABELS = 10
HIDDEN_LAYERS = [28]
MAX_ITERATIONS = 100

files = {'train_images': 'data/train-images.idx3-ubyte',
         'train_labels': 'data/train-labels.idx1-ubyte',
         'test_images': 'data/t10k-images.idx3-ubyte',
         'test_labels': 'data/t10k-labels.idx1-ubyte'}
files = {k: utils.format_training(
    idx2numpy.convert_from_file(v)) for k, v in files.items()}

f = lambda x: files[x]

size = len(f('train_images'))

train_images = f('train_images')[:int(m_*size)]
train_labels = f('train_labels')[:int(m_*size)]

test_images = f('test_images')
test_labels = f('test_labels')

def create_and_train():
    neural = NeuralNetwork(NUM_INPUTS,
                           NUM_LABELS)
    neural.set_regularization(REGULARIZATION)
    # me trying to cheese the scipy optimize.minimize learning rate by multiplying each gradient by this scaled factor
    neural.set_learning_rate(LEARNING_RATE)
    for layer in HIDDEN_LAYERS:
        neural.add_layer(layer)

    print('Training...')
    t1 = time.time()
    neural.train(train_images,
                 train_labels,
                 MAX_ITERATIONS)
    s = time.time()-t1
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    neural.plot_loss()
    print('Time elapsed: {0} hrs, {1} mins, {2} secs'.format(h, m, s))
    return neural

def main():
    """
    No cross-validation set, kind of lazy :)
    """
    neural = create_and_train()

    print('Testing training data...')
    train_error = neural.test(train_images,
                                train_labels)


    print('Testing testing data...')
    test_error =  neural.test(test_images,
                                test_labels)

    #compiling learning curve data
    """
    with open('data.pickle','rb') as p:
        data = pickle.load(p)
    data.append({'learning_rate': LEARNING_RATE,'regularization': REGULARIZATION,'training_size': int(m_*size), 'train_error': train_error, 'test_error': test_error})
    with open('data.pickle','wb') as p:
        pickle.dump(data,p)
    """

    #save model if it is good
    save_yn = input('Save? ([y]/n): ').lower()
    if save_yn == 'y':
        with open('model.pickle','wb') as p:
            pickle.dump(neural,p)

if __name__ == '__main__':
    main()
