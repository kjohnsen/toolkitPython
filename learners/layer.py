from neuron import OutputNeuron, Neuron

class Layer(object):
    def __init__(self, n, lr, add_bias=True):
        self.lr = lr
        self.nodes = []
        for i in range(n):
            self.nodes.append(Neuron(lr))
        # bias node
        self.nodes.append(Neuron)

class OutputLayer(Layer):
    def __init__(self, lr):
        super().__init__(0, lr, False)
        self.nodes = [OutputNeuron(lr)]
