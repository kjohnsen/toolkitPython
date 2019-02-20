import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import numpy as np
import random

from toolkit.supervised_learner import SupervisedLearner

class BackpropLearner(SupervisedLearner):
    lr = 0.1
    
    def __init__(self):
        self.hidden_layers = 1