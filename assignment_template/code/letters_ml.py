#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 00:32:05 2014

@author: alex
"""

import numpy as np
import sklearn
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot
import time

class ScalabilityQuestion(object):
    def __init__(self):
        self.prepare_data()
        
    def prepare_data(self):
        # Read the data from file
        data = np.genfromtxt('data/letter-recognition/letter-recognition.data',
                             delimiter=',')
        self.data = np.delete(data, 0, 1)
        # The first column of data is of type string, amd wasn't read properly
        data_raw = np.genfromtxt('data/letter-recognition/letter-recognition.data',
                                 delimiter=',', dtype=None)
        letters = list()
        for i in xrange(data_raw.size):
            letters.append(data_raw[i][0])        
        letters = np.array(letters)
        # Now encode the strings 
        self.le = sklearn.preprocessing.LabelEncoder()
        self.le.fit(letters)
        self.encoded_letters = self.le.transform(letters)[:,np.newaxis]
        
        
