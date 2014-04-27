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


def length(point1, point2):
    return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)


class tsp:
    """Travelling salesman solver: K-opt"""
    def __init__(self, points, cycles = 1000, kmax = 10, max_tour = 0.0,\
                 retries = 500):
        self.cycles = cycles
        self.kmax = kmax
        self.retries = retries
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

        # Find initial tour
        self._load_cached_path\
            ('/home/noah/class/discopt/tsp/testing/solutions/threeOpt_random_574.data')
        # self._find_greedy_path()
        # self._graphify()
        # self._random_start()
        self._run_kopt()


    def _load_cached_path(self, filename):
        """Load cached data if available"""
        fp = open(filename, 'r')
        data = ''.join(fp.readlines())
        fp.close()
        lines = data.split()
        self.path = map(int, lines[1:])
        for i in range(len(self.path) - 1):
            n1 = self.path[i]
            n2 = self.path[i+1]
            self.G.add_edge(n1, n2,\
                            weight = length(self.points[n1], self.points[n2]))
        n1 = self.path[0]
        self.G.add_edge(n2, n1, weight = length(self.points[n2], self.points[n1]))


    def _random_start(self):
        """Choose random initial path: too expensive to calculate
        distance matrix for very large inputs"""
        path = random.sample(range(self.N), self.N)
        for i in range(len(path)-1):
            p1 = path[i]
            p2 = path[i+1]
            self.G.add_edge(p1, p2,\
                            weight = length(self.points[p1], self.points[p2]))
        self.G.add_edge(p2, path[0], weight = length(self.points[p2], self.points[path[0]]))


    def _graphify(self):
        """Convert to graph data structure"""
        for e in range(len(self.edges)):
            c1, c2 = self._cities_from_edge(e)
            self.G.add_edge(c1, c2, weight = self.edges[e])


    def _cities_from_edge(self, edge):
        """Returns the two cities forming and edge"""
        return (self.path[edge], self.path[(edge + 1) % self.N])


    def _run_kopt(self):
        """Wrapper for _kopt"""
        count = 0
        best = self._tour_length()
        new_best = INF
        # print best
        rand_choice = False
        while count < self.cycles and best > self.max_tour:
            if new_best == best:
                if rand_choice:
                    for n in self.G.nodes():
                        self._kopt(st = n, rand=True)
                    rand_choice = False
                else: 
                    for n in self.G.nodes():
                        self._kopt(st = n)
                    rand_choice = True
            else:
                self._kopt(longest = True)
            new_best = self._tour_length()
            if new_best < best:
                print new_best
                self.save()
                best = new_best
            print count
            count += 1


    def _kopt(self, longest = False, st = None, rand=False):
        """swap k edges and choose best sequence of swaps"""
        ## Initialize vars
        added = {}
        removed = {}
        kvals = []
        k = 0
        pred = None
        k_init = self._tour_length()
        
        # pick starting edge
        if st:
            e = (st, st)
        elif longest:
            e = self._longest_edge()
        else:
            e = random.choice(self.G.edges())

        n = e[0]
        # swap edges k times
        while k < self.kmax:
            if rand:
                nn = None
                while not nn:
                    nn = random.choice(range(self.N))
                    if nn == n or n == pred: nn = None
            else:
                seen = [pred]
                nn = self._nearest_neighbor(n, seen)                    
                # while self.G.has_edge(n, nn):
                #     seen.append(nn)
                #     nn = self._nearest_neighbor(n, seen)

            np = self.G.successors(n)[0]
            nnp = self.G.successors(nn)[0]
            self.G.remove_edges_from([(n, np), (nn, nnp)])
            self.G.add_edges_from([(n, nn, {'weight' : self._extract_dist(n, nn)}),
                                (np, nnp, {'weight' : self._extract_dist(np, nnp)})])
            self._redirect(nn, np)

            # update variables
            added[k] = [(np, nnp), (n, nn)]
            removed[k] = [(n, np), (nn, nnp)]
            kvals.append(self._tour_length())
            k += 1
            pred = n
            n = nn

        # Find max kval and revert to that point
        if not kvals:
            return 
        best = min(range(len(kvals)), key = kvals.__getitem__)
        if kvals[best] > k_init:
            best = -1
        k = len(kvals)-1
        while k > best:
            self._remove_edges(added[k])
            for e in removed[k]:
                self.G.add_edge(e[0], e[1], weight = self._extract_dist(e[0], e[1]))
            e1 = removed[k][0][1]
            e2 = removed[k][1][0]
            self._redirect(e1, e2)
            k -= 1

        # update new path data
        self.path_length = self._tour_length()
        self.path = self._update_path()


    def _remove_edges(self, es):
        """Wrapper to remove edges, but if edge doesnt exist in one direction,
        try to remove it in the other direction"""
        for e in es:
            if self.G.has_edge(e[0], e[1]):
                self.G.remove_edge(e[0], e[1])
            else:
                self.G.remove_edge(e[1], e[0])


    def show(self, es = [], ns = []):
        """Produce image of the graph, optionally coloring specified
        nodes and edges"""
        pos = {i:self.points[i] for i in range(len(self.points))}
        if es:
            ecols = ['black'] * len(self.G.edges())
            for i in es:
                ecols[i] = 'r'
        if ns:
            ncols = ['r'] * len(self.G.edges())
            for i in ns:
                ncols[i] = 'b'
        if es and ns:
            nx.draw_networkx(self.G, edge_color = ecols, node_color = ncols, pos = pos)
        elif es:
            nx.draw_networkx(self.G, edge_color = ecols, pos = pos)
        elif ns:
            nx.draw_networkx(self.G, node_color = ncols, pos = pos)
        else:
            nx.draw_networkx(self.G, pos)
        plt.show()


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
        """Redirect graph between nodes of degree 0 and degree 2,
        tries given nodes first, but if they dont match degree it will
        search for correct nodes"""
        if self.G.out_degree(n1) == 0 and self.G.out_degree(n2) == 2:
            current = prev = n1
            end = n2
        elif self.G.out_degree(n2) == 0 and self.G.out_degree(n1) == 2:
            current = prev = n2
            end = n1
        else:
            start = []
            end = []
            for n in self.G.nodes():
                if self.G.out_degree(n) == 0:
                    start.append(n)
                if self.G.out_degree(n) == 2:
                    end.append(n)
            if len(start) == 0 and len(end) == 0:
                return
            assert len(start) == 1 and len(end) == 1, "start: {}, end {}".format(start,end)
            current = prev = start.pop()
            end = end.pop()
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


    def _survey_greedy_starts(self):
        """Run greedy solver and quick pass 3-opt from 
        all the nodes to find best starting node"""
        start_nodes = list(range(self.N))
        best_start = 0
        best_plen = INF
        for n in start_nodes:
            self.G = nx.DiGraph()
            self._find_greedy_path(st = n)
            # print "path after greedy: {}".format(self.path_length)
            self._graphify()
            self._three_opt()
            # print "path: {}".format(self.path_length)
            if self.path_length < best_plen:
                best_plen = self.path_length
                best_start = n
        # print "Best length: {}, starting from node: {}".\
        #     format(best_plen, best_start)
        return best_start


    def _run_three_opt(self):
        best = self.path_length
        while 1:
            self._three_opt()
            if self.path_length == best:
                break
            else:
                best = self.path_length
        if best > self.max_tour:
            count = 0
            while self.path_length > self.max_tour\
                  and count < self.retries:
                self._three_opt(longest = False)
                count += 1


    def _three_opt(self, longest = True):
        """Swap two edges in graph to search for better configuration"""
        tour = self._tour_length()
        num_edge = self.G.number_of_edges()
        G = self.G.copy()
        if longest:
            e1 = self._longest_edge()
        else:
            ind = random.choice(range(num_edge))
            e1 = self.G.edges()[ind]
        # print "Longest edge is: {}".format(e1)
        for i in range(num_edge):
            e2 = self.G.edges()[i]
            # print e2
            if e2 == e1 or e2[0] in e1 or e2[1] in e1:
                continue
            # Make first swap
            assert e1 in self.G.edges() and e2 in self.G.edges()
            self._swap_edges(e1, e2)
            opt1 = self._tour_length()
            opt2 = INF
            if opt1 < tour + self.wiggle:
                # print "swapped: {}, {}".format(e1, e2)
                self._redirect(e1[1], e2[0])
                # self.show()

                for j in range(num_edge):
                    e3 = self.G.edges()[j]
                    if e3 == e2 or e3 == e1 or e3[0] in e2 or e3[0] in e1\
                       or e3[1] in e2 or e3[1] in e1:
                        continue
                    # Make second swap
                    assert e3 in list(self.G.edges())
                    e4 = (e1[0], e2[0])
                    if e4 not in self.G.edges():
                        e4 = (e2[0], e1[0])
                    # print "swapping: {} with {}".format(e4, e3)
                    self._swap_edges(e4, e3)
                    opt2 = self._tour_length()

                    if opt2 < tour:
                        # print "swapped: {} with {}".format(e4, e3)
                        # self.show(es = [e1[0],e2[0],e3[0],e4[0]])
                        self._redirect(e4[1], e3[0])
                        break
                    # revert if old config better
                    self._swap_edges((e4[0], e3[0]), (e4[1], e3[1]))

            if opt2 < tour or opt1 < tour:
                # print "Improved to: {}".format(self._tour_length())
                break
            # revert if old config better
            self.G = G.copy()
            # assert self._tour_length() == tour
        # update new path data
        self.path_length = self._tour_length()
        self.path = self._update_path()

    def save(self):
        """save path for visualization"""
        filename = '/home/noah/class/discopt/tsp/testing/solutions/' +\
                   'kopt_' + str(self.N) + '.data'
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

    def _find_greedy_path(self, st = None, rand = False):
        """Find greedy path, taking shortest distance at each iteration"""
        self.G = nx.DiGraph()
        self.path = []
        self.path_length = 0.0
        self.edges = []
        start = 0
        if st:
            start = st
            # print "Starting from node {}".format(start)
        elif rand:
            start = random.choice(range(self.N))
        self.path.append(start)
        current = start
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

    # p1: 430, p2: 20800, p3: 30000, p4: 37600, p5: 323000
    T = tsp(points, kmax = 30, cycles = 4, max_tour = 37600)
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

