import abc
import numpy as np
import math

class Neuron(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, lr):
        self.lr = lr
        self.output = None
        self.delta = None
        self.input_nodes = []
        self.output_nodes = []
        self.input_weights = []
        self.output_weights = []
        self.forward_calc = True  # set to true while going forward, false while going back for checking
    
    def calc_set_out(self):
        self.forward_calc = True
        self.output = 1/(1+math.exp(-self.net))
        return self.output

    @property
    def net(self):
        inputs = [n.output for n in self.input_nodes]
        weighted_in = np.multiply(inputs, self.input_weights)
        return np.sum(weighted_in)

    @property
    def dout_dnet(self):
        return self.output * (1 - self.output)

    @abc.abstractmethod
    def calc_set_delta(self):
        self.forward_calc = False
        deltas = [n.delta for n in self.output_nodes]
        sum = np.sum(np.multiply(deltas, self.output_weights))
        self.delta = sum * self.dout_dnet
        return self.delta

    def weight_change_for_node(self, i):
        assert self.forward_calc is False
        return self.lr * self.input_nodes[i].output * self.delta


class OutputNeuron(Neuron):
    def __init__(self, lr):
        super().__init__(lr)
        self.target = None

    def calc_set_delta(self):
        self.forward_calc = False
        self.delta = (self.target - self.output) * self.dout_dnet
        return self.delta


class BiasNeuron(Neuron):
    def __init__(self, lr):
        super().__init__(lr)
        self.output = 1

    def calc_set_out(self):
        return 1


# class HiddenNeuron(Neuron):
#     def calc_set_delta(self):
#         self.forward_calc = False
#         deltas = [n.delta for n in self.output_nodes]
#         sum = np.sum(np.multiply(deltas, self.output_weights))
#         self.delta = sum * self.dout_dnet
#         return self.delta