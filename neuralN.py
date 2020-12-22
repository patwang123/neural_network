"""
sigmoid

"""
from scipy import optimize
import utils
import random
import numpy as np
import math


class NeuralNetwork:
    init_epsilon = 0.01
    def __init__(self,features,outputs,lambda_=0.0):
        self.layers = [features]
        self.thetas = []
        self.outputs = outputs
        self.features = features
        self.lambda_ = lambda_
    def set_regularization(self,lambda_ = 0.0):
        self.lambda_ = lambda_
    def add_layer(self,num):
        self.thetas.append(utils.initiate_thetas((num,self.layers[-1]+1),self.init_epsilon))
        self.layers.append(num)
    def train(self,features,labels,iters):
        self.add_layer(self.outputs)
        self.theta_dims = [t.shape for t in self.thetas]
        X = features
        y = labels
        options = {'maxiter': iters}
        lambda_ = 1
        costFunction = lambda p: self.cost_function(p,
                                                    X,
                                                    y)
        res = optimize.minimize(costFunction,
                                utils.flatten(self.thetas),
                                jac=True,
                                method='TNC',
                                options=options)
        self.thetas = utils.unflatten(res.x,self.theta_dims)

    def cost_function(self,thetas,features,labels):
        thetas = utils.unflatten(thetas,self.theta_dims)
        gradients = [np.zeros(t.shape) for t in thetas]
        num_layers = len(self.layers)
        J = 0
        m = features.shape[0]
        for i in range(m):
            x,y = features[i],[int(o == labels[i]) for o in range(self.outputs)]
            a = utils.forward_propagate(thetas,x)
            J += sum(y * np.vectorize(math.log)(a[-1]) + np.vectorize(lambda z: (1-z)*math.log(1-z))(a[-1]))
            errors = [None]*3
            errors[num_layers-1] = a[-1] - y
            for layer in range(num_layers-2,0,-1):
                errors[layer] = ((thetas[layer].T @ errors[layer+1]) * np.vectorize(lambda x: x*(1-x))(a[layer]))[1:]
            for layer in range(num_layers-2,-1,-1):
                err = errors[layer+1]
                a_l = a[layer]
                gradients[layer] += err.reshape((len(err),1)) @ a_l.reshape((1,len(a_l)))
        for i in range(len(gradients)):
            gradients[i] = 1/m * (gradients[i] + lambda_ * np.concatenate((np.zeros((thetas[i].shape[0], 1)), thetas[i][:, 1:]), axis=1))
        J *= -1/m
        J += self.lambda_/(2*m)*sum([np.linalg.norm(t[:,1:])**2 for t in thetas])
        gradients = np.concatenate([a.ravel() for a in gradients])
        return J, gradients
    def test(self,features,labels):
        pass
    def predict(self, features):
        pass
