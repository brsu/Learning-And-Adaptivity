#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Alex Moriarty
"""
import numpy as np
import sklearn
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot

# hint: Python 1d arrays, vs row and column vectors
# A = np.arange(10)
# print A.shape
# print A[:, np.newaxis]
# print A[np.newaxis, :]

class SimplePredictionQuestion(object):
    
    def draw_graph(self, classifier, filename="out.svg"):
        dot_data = StringIO()    
        tree.export_graphviz(classifier, out_file=dot_data)
        graph = pydot.graph_from_dot_data(dot_data.getvalue())
        graph.write_svg(filename+".svg")

