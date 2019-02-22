import random
from abc import ABC

from .neuron import *


class Connection(object):
    def __init__(self):
        self.weight = random.uniform(0.1, 0.5)

    def add_weight_change(self, weight_change):
        self.weight += weight_change


class Layer(ABC):
    def fully_connect_layer(self, next_layer, next_has_bias=True):
        if not next_has_bias:
            raise NotImplementedError()
        for j in self.nodes:
            for k in next_layer.nodes[:-1]:
                cnxn = Connection()
                j.output_nodes.append(k)
                j.output_cnxns.append(cnxn)
                k.input_cnxns.append(cnxn)
                k.input_nodes.append(j)
    
    def update_input_weights(self, prev_layer):
        for j in range(len(self.nodes) - 1):  # don't worry about bias node
            jnode = self.nodes[j]
            for i in range(len(prev_layer.nodes)):
                inode = prev_layer.nodes[i]
                weight_change_ij = jnode.weight_change_for_node(i)
                jnode.input_cnxns[i].add_weight_change(weight_change_ij)
                inode.output_cnxns[j].add_weight_change(weight_change_ij)

    def calc_deltas(self):
        for node in self.nodes:
            node.calc_set_delta()



class NonInputLayer(Layer):
    def calc_set_out(self):
        for n in self.nodes:
            n.calc_set_out()
            # print(f'non-input out = {n.output}')



class HiddenLayer(NonInputLayer):
    def __init__(self, n, lr, add_bias=True):
        self.lr = lr
        self.nodes = []
        for i in range(n):
            self.nodes.append(Neuron(lr))
        # bias node
        self.nodes.append(BiasNeuron(lr))

    def __repr__(self):
        node_chars = ['H' for i in range(len(self.nodes) - 1)]
        node_chars.append('B')
        return ' '.join(node_chars)


class OutputLayer(NonInputLayer):
    def __init__(self, num_classes, lr):
        self.lr = lr
        self.nodes = [OutputNeuron(lr) for i in range(num_classes)]

    @property
    def output_vec(self):
        return [n.output for n in self.nodes]

    @property
    def target_vec(self):
        return [n.target for n in self.nodes]

    def set_target(self, target):
        for i in range(len(self.nodes)):
            node = self.nodes[i]
            if i == target:
                node.target = 1
            else:
                node.target = 0
        
    def __repr__(self):
        node_chars = ['O' for i in range(len(self.nodes))]
        return ' '.join(node_chars)


class InputLayer(Layer):
    def __init__(self, num_feats, lr):
        self.lr = lr
        self.nodes = []
        for i in range(num_feats):
            self.nodes.append(InputNeuron(lr))
        self.nodes.append(BiasNeuron(lr))

    def set_inputs(self, inputs):
        for i in range(len(inputs)):
            self.nodes[i].set_out(inputs[i])

    def __repr__(self):
        node_chars = ['I' for i in range(len(self.nodes) - 1)]
        node_chars.append('B')
        return ' '.join(node_chars)
    