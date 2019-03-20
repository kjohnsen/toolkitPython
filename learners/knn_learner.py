from __future__ import (absolute_import, division, print_function, unicode_literals)
from collections import Counter
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import math
import copy
import heapq
from collections import Counter, defaultdict
import numpy as np
from operator import itemgetter

from toolkit.supervised_learner import SupervisedLearner
from toolkit.matrix import Matrix


class KNNLearner(SupervisedLearner):
    def train(self, features, labels):
        if labels.value_count(0) == 0:
            self.regression = True
            best_acc = float('inf')
        else:
            self.regression = False
            best_acc = 0
        self.cont_columns = []
        self.nom_columns = []
        for i in range(features.cols):
            if features.value_count(i) == 0:
                self.cont_columns.append(i)
            else:
                self.nom_columns.append(i)

        best_indices = range(features.rows)
        for i in range(3):
            indices = np.random.choice(features.rows, int(features.rows * 1), replace=False)
            self.setup(features, labels, indices)
            acc = self.measure_accuracy(features, labels)
            if self.accuracy_improved(acc, best_acc):
                best_acc = acc
                best_indices = indices
        self.setup(features, labels, best_indices)


    def setup(self, features, labels, reduction_indices):
        self.inst_array = np.array(features.data)[reduction_indices, :]
        self.train_labels = itemgetter(*reduction_indices)(labels.data)
        self.cont_array = self.inst_array[:, self.cont_columns]
        self.nom_array = self.inst_array[:, self.nom_columns]


    def accuracy_improved(self, current, best):
        if self.regression:
            return current < best
        else:
            return current > best

    def predict(self, features, labels):
        del labels[:]

        # calculate distance with all points, keep closest k
        k = 7
        h = []
        # distances = self.euclidean_distance(self.inst_array, features)
        distances = self.heom(features)

        indices = np.argpartition(distances, k)[:k]
        # print(indices)
        dist_labels = [(distances[i], self.train_labels[i][0]) for i in indices]

        vote = self.get_weighted_vote(dist_labels)
        # print(vote)
        labels.append(vote)

    def euclidean_distance(self, inst_array, novel_inst):
        dist_array = inst_array - novel_inst
        dist_array[np.logical_not(np.isfinite(dist_array))] = 1
        dist_array = np.sqrt(np.sum(np.square(dist_array), axis=1))
        return dist_array

    def heom_nom_distance(self, inst_array, novel_inst):
        dist_array = inst_array - novel_inst
        dist_array[np.logical_not(np.isfinite(dist_array))] = 1
        return np.sum(dist_array != 0, axis=1, dtype=np.int16)

    def heom(self, novel_inst):
        novel_inst = np.array(novel_inst)
        # get distances from continuous attributes
        const_inst = novel_inst[self.cont_columns]
        distances = self.euclidean_distance(self.cont_array, const_inst)

        # add distance from nominal attributes
        nom_inst = novel_inst[self.nom_columns]
        distances += self.heom_nom_distance(self.nom_array, nom_inst)
        return distances
        


    def get_unweighted_vote(self, nearest_neighbors):
        votes = [label for dist, label in nearest_neighbors]
        if not self.regression:
            counter = Counter(votes)
            result, num_votes = counter.most_common(1)[0]
        else:
            result = np.average(votes)
        return result

    def get_weighted_vote(self, nearest_neighbors):
        result = defaultdict(lambda: 0)
        if not self.regression:
            for dist, label in nearest_neighbors:
                if dist == 0: 
                    result[label] += float('inf')
                else:
                    result[label] += 1/(dist*dist)
            result = max(result.items(), key=itemgetter(1))[0]
        else:
            total_weight = 0
            result = 0
            for dist, output in nearest_neighbors:
                total_weight += 1/(dist*dist)
                result += output/(dist*dist)
            result /= total_weight
            
        return result
