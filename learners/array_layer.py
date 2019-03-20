import random
from abc import ABC
import numpy as np
import math

MOMENTUM = 0.5

class Layer(ABC):
    def fully_connect_layer(self, next_layer):
        self.next_layer = next_layer
        next_layer.prev_layer = self
        weights = np.zeros((self.num_nodes, next_layer.num_nb_nodes))
        for row in range(len(weights)):
            for col in range(len(weights[0])):
                weights[row, col] = random.uniform(0.1, 0.5)
        self.output_weights = weights
        next_layer.input_weights = weights
    
    def update_input_weights(self, prev_layer):
        try:
            # Start weight changes with momentum from previous weight changes
            if MOMENTUM:  # MOMENTUM
                self.weight_changes = np.multiply(self.weight_changes, MOMENTUM)
                self.weight_changes += np.matmul(prev_layer.out, self.delta[0:self.num_nb_nodes].T) * self.lr
            else:
                self.weight_changes = np.matmul(prev_layer.out, self.delta[0:self.num_nb_nodes].T) * self.lr
            new_weights = self.input_weights + self.weight_changes
            self.input_weights = new_weights
            prev_layer.output_weights = new_weights
        except AttributeError:  # if self.weight_changes isn't there
            self.weight_changes = np.zeros((len(prev_layer.out), self.num_nb_nodes))


def sigmoid(net):
    
    try:
        return 1/(1+math.exp(-net))
    except OverflowError:
        return float('inf')

sigmoid_vec = np.vectorize(sigmoid)
def sig_prime(out):
    return out*(1-out)


class HiddenLayer(Layer):
    def __init__(self, n, lr):
        self.lr = lr
        self.has_bias = True
        self.num_nodes = n+1
        self.num_nb_nodes = n  # non-bias nodes
        self.out = np.zeros((self.num_nodes, 1))
        self.out[self.num_nb_nodes, 0] = 1

    def calc_deltas(self):
        nl = self.next_layer
        self.delta = np.matmul(self.output_weights, nl.delta[0:nl.num_nb_nodes])
        self.delta = np.multiply(self.delta, sig_prime(self.out))

    def __repr__(self):
        node_chars = ['H' for i in range(self.num_nb_nodes)]
        node_chars.append('B')
        return ' '.join(node_chars)

    def calc_set_out(self):
        net = np.matmul(self.prev_layer.out.T, self.input_weights)
        self.out[0:self.num_nb_nodes, 0] = sigmoid_vec(net)


class OutputLayer(Layer):
    def __init__(self, num_classes, lr):
        self.lr = lr
        self.has_bias = False
        self.num_nodes = num_classes
        self.num_nb_nodes = num_classes  # non-bias nodes
        self.out = np.zeros((num_classes, 1))
        self.delta = np.zeros((num_classes, 1))
        self.target = np.zeros((num_classes, 1))

    def set_target(self, target):
        self.target = np.zeros((self.num_nodes, 1))
        self.target[int(target), 0] = 1

    def calc_set_out(self):
        net = np.matmul(self.prev_layer.out.T, self.input_weights).T
        self.out = sigmoid_vec(net)
        
    def calc_deltas(self):
        self.delta = np.multiply((self.target - self.out), sig_prime(self.out))

    def __repr__(self):
        node_chars = ['O' for i in range(self.num_nodes)]
        return ' '.join(node_chars)


class InputLayer(Layer):
    def __init__(self, num_feats, lr):
        self.lr = lr
        self.has_bias = True
        self.num_nodes = num_feats + 1
        self.num_nb_nodes = num_feats  # non-bias nodes
        self.out = np.zeros((self.num_nodes, 1))
        self.out[self.num_nb_nodes, 0] = 1

    def set_inputs(self, inputs):
        self.out[0:self.num_nb_nodes, 0] = np.array(inputs).T

    def __repr__(self):
        node_chars = ['I' for i in range(self.num_nb_nodes)]
        node_chars.append('B')
        return ' '.join(node_chars)
    