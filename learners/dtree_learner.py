from __future__ import (absolute_import, division, print_function, unicode_literals)
from collections import Counter
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import math
import copy
from operator import attrgetter
from collections import defaultdict

from toolkit.supervised_learner import SupervisedLearner
from toolkit.matrix import Matrix


class DTreeLearner(SupervisedLearner):
    def train(self, features, labels):
        vs_size = int(0.8 * features.rows)
        vs_features = Matrix(features, 0, 0, vs_size, features.cols)
        vs_labels = Matrix(labels, 0, 0, vs_size, labels.cols)
        print(f'Holding out {vs_size} instances for validation set')
        train_features = Matrix(features, vs_size, 0, features.rows-vs_size, features.cols)
        train_labels = Matrix(labels, vs_size, 0, labels.rows-vs_size, labels.cols)

        # self.build_tree_for_instances(train_features, train_labels)
        self.build_tree_for_instances(features, labels)
        # print(self.num_nodes)
        self.prune_tree_using_instances(vs_features, vs_labels)
        print(self.num_nodes)
        print(self.root.traverse())

        print(self.root)

    def build_tree_for_instances(self, features, labels):
        # make parent node
        self.root = Node(1, labels.most_common_value(0))
        # make unknown values a value
        for j in range(features.cols):
            self.unk_enum = len(features.enum_to_str[j].values())
            seen_unkown = False
            for i in range(features.rows):
                if features.get(i, j) == float('inf'):
                    # only set up if unknown exists in column
                    if not seen_unkown:
                        features.str_to_enum[j]['?'] = self.unk_enum
                        features.enum_to_str[j][self.unk_enum] = '?'
                        seen_unkown = True
                    features.set(i, j, self.unk_enum)
        # split on parent node
        self.root.split(features, labels)

    def prune_tree_using_instances(self, features, labels):
        prev_acc = 0
        this_acc = self.measure_accuracy(features, labels)
        while True:
            this_acc = -1
            for node in self.get_all_nodes(self.root):
                if node.output is not None: continue  # skip leaf nodes
                node.prune()
                acc = self.measure_accuracy(features, labels)
                if acc > this_acc:
                    this_acc = acc
                    node_to_prune = node
                node.unprune()
            if this_acc >= prev_acc:
                prev_acc = this_acc
                node_to_prune.prune()
            else:
                break

    def predict(self, features, labels):
        del labels[:]

        features = [self.unk_enum if x == float('inf') else x for x in features]
        labels.append(self.root.get_class_for_row(features))

    def get_all_nodes(self, node):
        if node.output is not None:
            return [node]
        else:
            result = []
            for child in node.children.values():
                result += self.get_all_nodes(child)
            result += [node]
            return result

    @property
    def num_nodes(self):
        return len(self.get_all_nodes(self.root))

class Node():
    def __init__(self, weight, mode_class):
        self.weight = weight
        self.mode_class = mode_class
        self.children = {}
        self.output = None

    def __str__(self, margin=''):
        if self.output is not None: return str(self.output) + '\n'
        result = self.att_name + '?\n'
        margin += '  '
        for enum, child in self.children.items():
            result += margin + self.children_names[int(enum)] + ' --> ' + self.children[enum].__str__(margin)
        return result

    def traverse(self, decision_attributes_for_level=defaultdict(list), current_level=1):
        if self.output is not None: return
        decision_attributes_for_level[current_level].append(self.att_name)
        for c in self.children.values():
            c.traverse(decision_attributes_for_level, current_level+1)
        return decision_attributes_for_level

    def __repr__(self):
        if self.output is not None:
            return f'Leaf node with output class {self.output}'
        elif len(self.children) > 1:
            return f'{self.att_name} node with {len(self.children)} children'

    def split(self, features, labels):
        # if class is pure for node, stop
        if labels.count_values_present(0) == 1:
            self.output = labels.get(0, 0)
            return
        elif features.cols == 0:  # no more features to split on
            self.output = labels.most_common_value(0)
            return
        # find lowest info attribute (highest info gain)
        self.att_idx = self.highest_info_gain_att(features, labels)
        self.att_name = features.attr_name(self.att_idx)
        # print(f'Splitting on {self.att_name}')
        self.children_names = features.enum_to_str[self.att_idx]
        # split on each value of that attribute
        features_for_value = {}
        labels_for_value = {}
        # iterate over rows and build data for use in child nodes
        for i in range(features.rows):
            row = features.row(i)
            label = labels.row(i)
            value = row[self.att_idx]
            new_row = copy.deepcopy(row)
            del new_row[self.att_idx]
            try:
                features_for_value[value].data.append(new_row)
                labels_for_value[value].data.append(label)
            except KeyError:
                new_attr_names = copy.deepcopy(features.attr_names)
                del new_attr_names[self.att_idx]
                new_str_to_enum = copy.deepcopy(features.str_to_enum)
                del new_str_to_enum[self.att_idx]
                new_enum_to_str = copy.deepcopy(features.enum_to_str)
                del new_enum_to_str[self.att_idx]
                features_for_value[value] = Matrix()
                features_for_value[value].data = [new_row]
                features_for_value[value].str_to_enum = new_str_to_enum
                features_for_value[value].enum_to_str = new_enum_to_str
                features_for_value[value].attr_names = new_attr_names

                labels_for_value[value] = Matrix()
                labels_for_value[value].data = [label]
        for value in sorted(features_for_value.keys()):
            weight = features_for_value[value].rows / features.rows
            n = Node(weight, labels.most_common_value(0))
            # print(f'Creating child for {self.att_name} = {features.enum_to_str[self.att_idx][value]}')
            self.children[value] = n
            n.split(features_for_value[value], labels_for_value[value])

    def get_class_for_row(self, row):
        if self.output is not None: return self.output

        value = row[self.att_idx]
        new_row = row.copy()
        del new_row[self.att_idx]
        # print(f'Deleting element {self.att_idx}, value={value}')
        try:
            return self.children[value].get_class_for_row(new_row)
        except KeyError:
            mode_node = max(self.children.values(), key=attrgetter('weight'))
            return mode_node.get_class_for_row(new_row)


    def highest_info_gain_att(self, features, labels):
        highest_gain_ratio = 0
        best_att_idx = -1
        curr_info = self.calc_info([x[0] for x in labels.data])
        for att_idx in range(features.cols):
            # print(f'Calculating info for attribute {features.attr_name(att_idx)}')
            info_left, split_info = self.calc_info_left_and_si_for_att(att_idx, features, labels)
            info_gain = curr_info - info_left
            gain_ratio = info_gain/split_info if split_info != 0 else float('inf')
            # gain_ratio = info_gain
            # print(f'For attribute {features.attr_name(att_idx)}, info left would be {info_left}')
            if gain_ratio > highest_gain_ratio:
                highest_gain_ratio = gain_ratio
                best_att_idx = att_idx
        return best_att_idx

    def calc_info_left_and_si_for_att(self, att_idx, features, labels):
        # count labels by attribute values
        labels_for_value = {}
        for i in range(features.rows):
            value = features.get(i, att_idx)
            label = labels.get(i, 0)
            try:
                labels_for_value[value].append(label)
            except KeyError:
                labels_for_value[value] = [label]

        result = 0
        split_info = 0
        for l in labels_for_value.values():
            result += self.calc_info(l) * len(l) / features.rows
            split_info -= len(l)/features.rows * math.log2(len(l)/features.rows)
        return result, split_info
    
    def calc_info(self, labels):
        total = len(labels)
        result = 0
        counter = Counter(labels)
        for n in counter.values():
            result += -n/total * math.log2(n/total)
        return result

    def prune(self):
        self.output = self.mode_class

    def unprune(self):
        self.output = None






            


