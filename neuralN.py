"""
sigmoid

"""
from scipy import optimize
import utils
import random
import numpy as np
import math
import matplotlib.pyplot as plt

class NeuralNetwork:
    init_epsilon = 0.12
    def __init__(self,features,outputs,lambda_=0.0,adjusted_learning=1.0):
        self.layers = [features]
        self.thetas = []
        self.outputs = outputs
        self.features = features
        self.lambda_ = lambda_
        self.learning_rate = adjusted_learning
    
    def set_regularization(self,lambda_ = 0.0):
        self.lambda_ = lambda_

    def set_learning_rate(self,adjusted_learning=1.0):
        self.learning_rate = adjusted_learning

    def add_layer(self,num):
        self.thetas.append(utils.initiate_thetas((num,self.layers[-1]+1),self.init_epsilon))
        self.layers.append(num)

    def train(self,features,labels,iters):
        self.loss = []
        self.add_layer(self.outputs)
        self.theta_dims = [t.shape for t in self.thetas]
        X = features
        y = labels
        options = {'maxiter': iters}
        costFunction = lambda p: self.cost_function(p,
                                                    X,
                                                    y)
        self.iter = 0
        res = optimize.minimize(costFunction,
                                utils.flatten(self.thetas),
                                jac=True,
                                method='TNC',
                                options=options)
        self.thetas = utils.unflatten(res.x,self.theta_dims)

    def plot_loss(self):
        y = self.loss
        x = list(range(len(self.loss)))
        plt.scatter(x,y)
        plt.show()

    def cost_function(self,thetas,features,labels):
        self.iter += 1
        print(self.iter, end='... ')
        thetas = utils.unflatten(thetas,self.theta_dims)
        gradients = [np.zeros(t.shape) for t in thetas]
        num_layers = len(self.layers)
        J = 0
        m = features.shape[0]
        logg = np.vectorize(math.log)
        one_minus = np.vectorize(lambda z: 1-z)
        for i in range(m):
            x,y = features[i],[int(o == labels[i]) for o in range(self.outputs)]
            a = utils.forward_propagate(thetas,x)
            J += sum(y * logg(a[-1]) + one_minus(y) * logg(one_minus(a[-1])))
            errors = [None]*num_layers
            errors[num_layers-1] = a[-1] - y
            for layer in range(num_layers-2,0,-1):
                errors[layer] = ((thetas[layer].T @ errors[layer+1]) * np.vectorize(lambda x: x*(1-x))(a[layer]))[1:]
            for layer in range(num_layers-2,-1,-1):
                err = errors[layer+1]
                a_l = a[layer]
                gradients[layer] += err.reshape((len(err),1)) @ a_l.reshape((1,len(a_l)))
        for i in range(len(gradients)):
            gradients[i] = self.learning_rate * 1/m * (gradients[i] + self.lambda_ * np.concatenate((np.zeros((thetas[i].shape[0], 1)), thetas[i][:, 1:]), axis=1))
        J *= -1/m
        J += self.lambda_/(2*m)*sum([np.linalg.norm(t[:,1:])**2 for t in thetas])
        print('loss: {0}'.format(J))
        gradients = np.concatenate([a.ravel() for a in gradients])
        self.loss.append(J)
        return J, gradients

    def test(self,features,labels):
        total = len(labels)
        count = 0
        for i in range(len(labels)):
            res = np.argmax(utils.forward_propagate(self.thetas,features[i])[-1])
            if res == labels[i]:
                count += 1
        print('Accuracy: {0}'.format(count/total))
        return (1-count)/total #error

    def predict(self, features):
        pass
