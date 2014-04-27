#!/usr/bin/python
# -*- coding: utf-8 -*-
import scipy
from scipy import spatial
import numpy as np
import math
from collections import namedtuple

Point    = namedtuple("Point", ['x', 'y'])
MAX_NODE = 100000
INF      = 1000000

def nCk(n, k):
    """n choose k"""
    f = math.factorial
    if n - k < 0:
        return 0
    return f(n) / f(k) / f(n - k)


def length(point1, point2):
    return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)


class tsp:
    """Greedy travelling salesman solver"""
    def __init__(self, points):
        self.points = points
        self.N = len(points)
        assert self.N <= MAX_NODE,\
            "Too many nodes, max currently set at {}".format(MAX_NODE)
        self.path = []
        self.dist = scipy.spatial.distance.pdist(points)
        self.path_length = 0.0

        # Find tour
        self._find_path()

    def save(self):
        """save path for visualization"""
        filename = '/home/noah/class/discopt/tsp/testing/solutions/' +\
                   'greedy_' + str(self.N) + '.data'
        fp = open(filename, 'w')
        fp.write("%s " % self.path_length)
        for p in self.path:
            fp.write("%s " % p)
        fp.close()


    def _extract_dist(self, i, j):
        """X[i, j] and X[j, i] values are set to 
        v[{n choose 2}-{n-i choose 2} + (j-i-1)]"""
        if i == j:
            return 0.0
        if i < j:
            i, j = j, i
        ind = (self.N * j) - (j * (j + 1) / 2) + (i - 1 - j)
        return self.dist[ind]


    def _index_min(self, values):
        return min(xrange(len(values)), key = values.__getitem__)


    def _nearest_neighbor(self, i, seen = []):
        """Returns the nearest neighbor to point i"""
        dists = []
        for j in xrange(self.N):
            if j == i or j in seen:
                dists.append(INF)
            else:
                dists.append(self._extract_dist(i, j))
        # print dists
        return self._index_min(dists)

    def _find_path(self):
        """Find greedy path, taking shortest distance at each iteration"""
        self.path.append(0)
        current = 0
        while (len(self.path) < self.N):
            next_stop = self._nearest_neighbor(current, self.path)
            self.path_length += self._extract_dist(current, next_stop)
            self.path.append(next_stop)
            current = next_stop
        self.path_length += self._extract_dist(0, self.path[-1])

    
    def _output(self):
        output_data = str(self.path_length) + ' ' + str(0) + '\n'
        output_data += ' '.join(map(str, self.path))
        return output_data


def solve_it(input_data):
    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])

    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    T = tsp(points)
    T.save()
    return T._output()


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

