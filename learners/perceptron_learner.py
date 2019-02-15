
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import numpy as np
import random

from toolkit.supervised_learner import SupervisedLearner
from learners.perceptron import Perceptron

class PerceptronLearner(SupervisedLearner):
    lr = 0.1
    def __init__(self):
        pass

    
    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """
        self.perceptrons = []

        for row in features.data:
            row.append(1)  # add bias input
        features.attr_names.append("bias")
        
        self.num_classes = labels.value_count(0)
        if self.num_classes == 2:
            percy = Perceptron(features.cols, self.lr)
            self.perceptrons.append(percy)
            self.train_perceptron(percy, features, labels)
        elif self.num_classes > 2:
            for class_enum in range(self.num_classes):
                class_specific_labels = labels.copy()
                # necessary so original features don't get shuffled separate from original labels
                features_copy = features.copy()
                for row in class_specific_labels.data:
                    if row[0] == class_enum:
                        row[0] = 1
                    else:
                        row[0] = 0
                percy = Perceptron(features.cols, self.lr)
                self.perceptrons.append(percy)
                self.train_perceptron(percy, features_copy, class_specific_labels)

    def train_perceptron(self, perceptron, features, labels):
        boring_epochs = 0
        epochs = 0
        best_correct = 0
        while boring_epochs < 10:
            correct = 0
            for i in range(features.rows):
                inputs = features.row(i)
                target = labels.row(i)[0]
                correct += perceptron.process_instance(inputs, target)

            if correct > best_correct:
                boring_epochs = 0
                best_correct = correct
            else: 
                boring_epochs += 1

            epochs += 1
            features.shuffle(labels)
            print(f'Accuracy for epoch = {correct/features.rows}')
            # misclassification rate
            # print(1-self.measure_accuracy(features, labels))

        # print(f'Final weight vector:')
        # for i in range(len(perceptron.weights)-1):
        #     feature = features.attr_name(i)
        #     print(f'{feature}: {perceptron.weights[i]}')
        print(f'{epochs} epochs elapsed in training')
        # print(f'Accuracy for perceptron: {self.measure_accuracy(features, labels)}')


    def predict(self, features, labels):
        del labels[:]
        inputs = features.copy()
        if len(inputs) == len(self.perceptrons[0].weights) - 1: inputs.append(1)

        if self.num_classes == 2:
            labels.append(self.perceptrons[0].calculate_output(inputs))

        elif self.num_classes > 2:
            highest_net = -float('inf')
            # should work for any number of classes
            for class_enum in range(len(self.perceptrons)):
                percy = self.perceptrons[class_enum]
                out, net = percy.calculate_output(inputs, net=True)
                # print(f'class: {class_enum}, net: {net}, out: {out}')
                if net > highest_net:
                    highest_net = net
                    winner = class_enum
            # print(f'highest net is {highest_net} for class {winner}')
            labels.append(winner)


