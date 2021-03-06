import random
import numpy as np

class Perceptron:
    def __init__(self, num_features, learning_rate):
        # last weight is bias
        self.weights = np.array([random.uniform(0.1, 0.5) for x in range(num_features)])
        self.l_rate = learning_rate

    # Returns 1 if it was correct, to keep track of accuracy
    def process_instance(self, inputs, target):
        out = self.calculate_output(inputs)
        # c*input*(target-out)
        weight_delta = [self.l_rate * input * (target - out) for input in inputs]
        self.weights += weight_delta

        if target == out: return 1
        else: return 0
        

    def calculate_output(self, inputs, net=False):
        # print(f"Inputs = {inputs}")
        # print(f'Weights = {self.weights}')
        weighted_in = np.multiply(inputs, self.weights)
        net_out = np.sum(weighted_in)
        if net_out > 0: out = 1
        else: out = 0

        if net: return (out, net_out)
        else: return out

