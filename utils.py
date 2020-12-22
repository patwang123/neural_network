import numpy as np
import math
import random

def sigmoid(z):
    sig = lambda z: 1/(1+math.exp(z))
    try:
        return np.vectorize(sig)(z)
    except:
        return sig(z)
def sigmoid_gradient(a):
    return np.vectorize(lambda i: i*(1-i))(a)

def initiate_thetas(dimensions,epsilon):
    arr = np.zeros(dimensions)
    return np.vectorize(lambda i: random.random()*2*epsilon - epsilon)(arr)

def flatten(arr):
    return np.concatenate([a.ravel() for a in arr],axis=0)

def unflatten(arr,theta_dims):
    res = []
    for dim in theta_dims:
        tbd = arr[:dim[0]*dim[1]]
        arr = arr[dim[0]*dim[1]:]
        res.append(tbd.reshape(dim))
    return res


def forward_propagate(thetas, x_row):
    theta_copy = thetas[:]
    add_one = lambda x: np.concatenate(([1], x), axis=0)
    a = [add_one(x_row[:].T)]
    for t in thetas:
        last = a[-1]
        new_a = sigmoid(theta_copy.pop(0) @ last)
        a.append(add_one(new_a))
    a[-1] = a[-1][1:]
    return np.array(a, dtype='object')
