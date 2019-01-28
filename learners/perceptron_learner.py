
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import numpy as np
import random

from toolkit.supervised_learner import SupervisedLearner
from learners.perceptron import Perceptron

class PerceptronLearner(SupervisedLearner):
    def __init__(self):
        pass

    
    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """
        for row in features.data:
            row.append(1)  # add bias input
        
        if labels.value_count(0) == 2:
            self.percy = Perceptron(2, 0.1)
            boring_epochs = 0
            best_correct = 0
            while boring_epochs < 10:
                correct = 0
                for i in range(features.rows):
                    inputs = features.row(i)
                    target = labels.row(i)[0]
                    correct += self.percy.process_instance(inputs, target)

                if correct > best_correct:
                    boring_epochs = 0
                    best_correct = correct
                    print(f'Correct = {correct}')
                else: 
                    boring_epochs += 1
                    # TODO: refactor to use measure_accuracy
        
        else:
            raise Exception("Non-binary classification not implemented")


    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        del labels[:]
        inputs = features.copy()
        if len(inputs) == len(self.percy.weights) - 1: inputs.append(1)
        labels.append(self.percy.calculate_output(inputs))

