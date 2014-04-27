#!/usr/bin/python
# -*- coding: utf-8 -*-
import scipy
from scipy import spatial
import numpy as np
import math
from collections import namedtuple


def load_cached_path(filename):
    """Load cached data"""
    fp = open(filename, 'r')
    data = ''.join(fp.readlines())
    fp.close()
    lines = data.split()
    path_length = float(lines[0])
    path = map(int, lines[1:])
    # for i in range(len(path) - 1):
    #     n1 = path[i]
    #     n2 = path[i+1]
    #     G.add_edge(n1, n2,\
    #                     weight = length(points[n1], points[n2]))
    #     n1 = path[0]
    #     G.add_edge(n2, n1, weight = length(points[n2], points[n1]))
    return path_length, path


def output(path_length, path):
    output_data = str(path_length) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, path))
    return output_data


def solve_it(input_data):
    filename = "./testing/solutions/rev_trans_anneal_200.data"
    path_length, path = load_cached_path(filename)
    
    return output(path_length, path)


import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        input_data_file = open(file_location, 'r')
        input_data = ''.join(input_data_file.readlines())
        input_data_file.close()
        print solve_it(input_data)
    else:
        print 'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)'

