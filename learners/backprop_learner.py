import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import numpy as np
import random

from .array_layer import *
from toolkit.supervised_learner import SupervisedLearner
from toolkit.matrix import Matrix


class BackpropLearner(SupervisedLearner):
    lr = 0.8
    validation_set_proportion = 0.14
    
    def train(self, features, labels):
        self.setup_network(features.cols, labels.value_count(0))
        vs_size = int(self.validation_set_proportion * features.rows)
        vs_features = Matrix(features, 0, 0, vs_size, features.cols)
        vs_labels = Matrix(labels, 0, 0, vs_size, labels.cols)
        print(f'Holding out {vs_size} instances for validation set')
        train_features = Matrix(features, vs_size, 0, features.rows-vs_size, features.cols)
        train_labels = Matrix(labels, vs_size, 0, labels.rows-vs_size, labels.cols)

        stagnant_epochs = 0
        epochs = 0
        best_vs_mse = float('inf')
        while stagnant_epochs < 40:
            this_train_mse = 0
            for i in range(train_features.rows):
                inputs = train_features.row(i)
                target = train_labels.row(i)[0]
                self.calc_set_out(inputs)
                self.output_layer.set_target(target)
                self.calc_deltas()
                self.update_weights()
                this_train_mse += self.se_for_instance()
            this_train_mse = this_train_mse / train_features.rows

            epochs += 1
            this_vs_mse, this_vs_accy = self.calc_mse_and_accy(vs_features, vs_labels)
            # print(f'{epochs},{this_train_mse},{this_vs_mse},{this_vs_accy}')

            if this_vs_mse < best_vs_mse:
                stagnant_epochs = 0
                best_vs_mse = this_vs_mse
            else:
                stagnant_epochs += 1

            train_features.shuffle(train_labels)

        self.write_out(f'{epochs},{self.hidden_layers[-1].num_nb_nodes},{this_train_mse},{this_vs_mse},')
        print(f'{epochs} epochs elapsed in training')


    def predict(self, features, labels):
        del labels[:]
        self.calc_set_out(features)
        labels.append(np.argmax(self.output_layer.out))  # returns index with highest number, which is the class

    def measure_accuracy(self, features, labels, confusion=None):
        if features.rows != labels.rows:
            raise Exception("Expected the features and labels to have the same number of rows")
        if labels.cols != 1:
            raise Exception("Sorry, this method currently only supports one-dimensional labels")
        if features.rows == 0:
            raise Exception("Expected at least one row")

        mse, accy = self.calc_mse_and_accy(features, labels)
        self.write_out(str(mse) + '\n')

        return accy

    def write_out(self, string):
        out_file = open('out.csv', 'a')
        out_file.write(string)
        out_file.close()

    def setup_network(self, num_feats, num_classes):
        self.input_layer = InputLayer(num_feats, self.lr)
        self.hidden_layers = [HiddenLayer((num_feats), self.lr)]
        # MULTIPLE LAYERS:
        # self.hidden_layers.append(HiddenLayer(10, self.lr))
        self.num_classes = num_classes
        self.output_layer = OutputLayer(self.num_classes, self.lr)
        self.layers = [self.input_layer] + self.hidden_layers + [self.output_layer]
        for l in self.layers[::-1]:
            print(l)
        self.connect_layers()

    def calc_mse_and_accy(self, features, labels):
            sse = 0
            num_correct = 0
            for i in range(features.rows):
                inputs = features.row(i)
                target = labels.row(i)[0]
                self.calc_set_out(inputs)
                self.output_layer.set_target(target)
                sse += self.se_for_instance()
                if np.argmax(self.output_layer.out) == target:
                    num_correct += 1
            return sse / features.rows, num_correct/features.rows
 

    def calc_set_out(self, inputs):
        self.input_layer.set_inputs(inputs)
        for hl in self.hidden_layers:
            hl.calc_set_out()
        return self.output_layer.calc_set_out()

    def calc_deltas(self):
        for non_input_layer in self.layers[:0:-1]:
            non_input_layer.calc_deltas()

    def se_for_instance(self):
        ol = self.output_layer
        result = ol.target - ol.out
        result = np.square(result)
        return np.sum(result)

    def update_weights(self):
        for j in range(len(self.layers) - 1, 0, -1):
            self.layers[j].update_input_weights(self.layers[j-1])

    def connect_layers(self):
        for i in range(len(self.layers) - 1):
            this_layer = self.layers[i]
            next_layer = self.layers[i+1]
            this_layer.fully_connect_layer(next_layer)