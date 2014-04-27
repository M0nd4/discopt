#!/usr/bin/python
# -*- coding: utf-8 -*-
import random
import networkx as nx
import scipy
from scipy import spatial
import numpy as np
import math
import matplotlib.pyplot as plt
# from loadData import *
from collections import namedtuple

Point    = namedtuple("Point", ['x', 'y'])
MAX_NODE = 100000
INF      = 100000000

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
    def __init__(self, points, max_tour = INF):
        self.max_tour = max_tour
        self.points = points
        self.N = len(points)
        assert self.N <= MAX_NODE,\
            "Too many nodes, max currently set at {}".format(MAX_NODE)
        self.path = []
        self.dist = scipy.spatial.distance.pdist(points)
        self.path_length = 0.0
        self.edges = []
        self.G = nx.DiGraph()

        # Find greedy tour
        self._find_greedy_path()
        self._graphify()
        self._run_two_opt()


    def show(self):
        """Produce image of the graph"""
        nx.draw(self.G)
        plt.show()


    def _graphify(self):
        """Convert to graph data structure"""
        for e in range(len(self.edges)):
            c1, c2 = self._cities_from_edge(e)
            self.G.add_edge(c1, c2, weight = self.edges[e])


    def _longest_edge(self):
        """Find the longest edge in the tour"""
        longest_edge = None
        max_wt = 0.0
        for e in self.G.edges():
            wt = self.G.edge[e[0]][e[1]]['weight']
            if wt > max_wt:
                max_wt = wt
                longest_edge = e
        return longest_edge


    def _tour_length(self):
        """Return the current tour length"""
        tour = 0.0
        for e in self.G.edges():
            tour += self.G.edge[e[0]][e[1]]['weight']
        return tour
    
    
    def _update_path(self):
        """Update the path along tour, works for directed or undirected graph"""
        stack = [0]
        path = [0]
        prev = 0
        done = False
        while (not done):
            node = stack.pop()
            # assert node not in path, "path update failure on node {}".format(node)
            opts = list(self.G[node])
            n = opts[0]
            if n == prev:
                n = opts[1]
            if n == 0:
                return path
            else:
                stack.append(n)
                path.append(n)
                prev = node
        self.path = path


    def _cities_from_edge(self, edge):
        """Returns the two cities forming and edge"""
        return (self.path[edge], self.path[(edge + 1) % self.N])

    def _swap_edges(self, e1, e2, config = 1):
        """Swap two edges in directed graph"""
        c1_t, c1_h, c2_t, c2_h = e1[0], e1[1], e2[0], e2[1]
        # Remove the two selected edges
        self.G.remove_edge(c1_t, c1_h)
        self.G.remove_edge(c2_t, c2_h)
        # Connect c1_t -> c2_t, c1_h -> c2_h
        self.G.add_edge(c1_t, c2_t, weight = self._extract_dist(c1_t, c2_t))
        self.G.add_edge(c1_h, c2_h, weight = self._extract_dist(c1_h, c2_h))
        # self._redirect(e1[1], e2[0])

    def _redirect(self, n1, n2):
        """Redirect directed graph between two nodes, breaks
        if there are redirections between nodes"""
        current = prev = n1
        end = n2
        if not self.G.predecessors(current):
            current = prev = n2
            end = n1
        while (current != end):
            pred = self.G.predecessors(current)[0]
            if pred == prev:
                pred = self.G.predecessors(current)[1]
            # add new reversed edge
            wt = self.G[pred][current]['weight']
            self.G.add_edge(current, pred, weight = wt)
            # remove edge going the other way
            self.G.remove_edge(pred, current)
            prev = current
            current = pred


    def _run_two_opt(self):
        best = self.path_length
        while 1:
            self._two_opt()
            if self.path_length == best:
                break
            else:
                best = self.path_length
        if best > self.max_tour:
            while self.max_tour < self.path_length:
                self._two_opt(longest = False)

    def _two_opt(self, longest = True):
        """Swap two edges in graph to search for better configuration"""
        tour = self._tour_length()
        num_edge = self.G.number_of_edges()
        if longest:
            e1 = self._longest_edge()
        else:
            ind = random.choice(range(num_edge))
            e1 = self.G.edges()[ind]
        # print "Longest edge is: ({}, {})".format(e1[0], e1[1])
        for i in range(num_edge):
            e2 = self.G.edges()[i]
            # print e2
            if e2 == e1 or e2[0] in e1 or e2[1] in e1:
                continue
            # try new config
            self._swap_edges(e1, e2)
            opt1 = self._tour_length()
            
            if opt1 < tour:
                self._redirect(e1[1], e2[0])
                break

            # revert if old config better
            self._swap_edges((e1[0], e2[0]), (e1[1], e2[1]))
            # assert self._tour_length() == tour

        self.path_length = self._tour_length()
        self.path = self._update_path()

    def save(self):
        """save path for visualization"""
        filename = '/home/noah/class/discopt/tsp/testing/solutions/' +\
                   '2opt_' + str(self.N) + '.data'
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

    def _index_max(self, values):
        return max(xrange(len(values)), key = values.__getitem__)


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

    def _find_greedy_path(self):
        """Find greedy path, taking shortest distance at each iteration"""
        self.path.append(0)
        current = 0
        while (len(self.path) < self.N):
            next_stop = self._nearest_neighbor(current, self.path)
            new_dist = self._extract_dist(current, next_stop)
            self.path_length += new_dist
            self.path.append(next_stop)
            self.edges.append(new_dist)
            current = next_stop
        dist_home = self._extract_dist(0, self.path[-1])
        self.path_length += dist_home
        self.edges.append(dist_home)
    
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

    T = tsp(points, 433)
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

