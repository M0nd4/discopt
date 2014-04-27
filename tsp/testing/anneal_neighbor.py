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
INF      = 1000000000


def length(point1, point2):
    return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def acceptance_probability(energy, new_energy, temp):
    """Calculates the probability of accepting a neighboring solution
    based on the current temperature and metropolis"""
    if new_energy < energy:
        return 1.0
    return np.exp(-abs(float(energy - new_energy)) / temp)


class tsp:
    """Travelling salesman solver for large sets where distance matrix isn't
    feasible, calculates neighborhoods and runs either kopt or simulated annealing
    in which both select neighbors from surrounding quadrats"""
    def __init__(self, points, cycles = INF, temp = 10, cooling_rate = 0.00005, 
                 max_tour = 0.0):
        self.cycles = cycles
        self.max_tour = max_tour
        self.points = points
        self.N = len(points)
        assert self.N <= MAX_NODE,\
            "Too many nodes, max currently set at {}".format(MAX_NODE)
        self.path = []
        # self.dist = scipy.spatial.distance.pdist(points)
        self.path_length = 0.0
        self.edges = []
        
        # Neighborhood variables
        self.dim = 4  # make 40 x 40 grid neighborhoods for large
        self.nsize = 2
        self.x_cuts, self.y_cuts = self._make_gridlines()
        self.hoods = self._construct_hoods()
        self._cache_neighbors()

        # Annealing
        self.temp = temp
        self.cooling_rate = cooling_rate

        # Graph
        self.G = nx.DiGraph()

        # Find initial tour
        self._load_cached_path\
            ('/home/noah/class/discopt/tsp/testing/solutions/anneal_51.data')
        # self._initial_solution()
        # self._graphify()
        self.path_length = self._tour_length_path(self.path)


    ####################################################################################
    ##                           Simulated Annealing
    def _run_anneal(self):
        """Wrapper for simulated annealing"""
        count = 0
        while count < self.cycles and self.path_length > self.max_tour:
            print "Count: {}".format(count)
            best = self.path_length
            self._anneal()

            if self.path_length < best:
                print "New best: {}\nSaving...".format(self.path_length)
                self.save()
            count += 1


    def _anneal(self):
        """cycle of simulated annealing"""
        current = list(self.path)
        temp = self.temp

        while temp > 1:
            new_path = list(current)
            c1 = np.random.randint(self.N)
            p1 = self.points[c1]

            nbrs = self.hoods[self._nhood(p1)]
            if len(nbrs) < 2:
                continue
            p2, c2 = random.choice( nbrs )
            while c1 == c2:
                p2, c2 = random.choice( nbrs )
            new_path[c1], new_path[c2] = new_path[c2], new_path[c1]
            
            # Calculate energies
            current_energy = self._tour_length_path(current)
            nbr_energy = self._tour_length_path(new_path)

            # decide to accept
            prob = np.random.random()
            if acceptance_probability(current_energy, nbr_energy, temp) > prob:
                current = new_path
                current_energy = nbr_energy

            # update best path
            if current_energy < self.path_length:
                print "Current tour: {}".format(current_energy)
                self.path_length = current_energy
                self.path = list(current)
            
            # update temp
            temp *= (1 - self.cooling_rate)


    ####################################################################################
    ##                           K-opt
    def _run_kopt(self):
        """Wrapper for _kopt"""
        count = 0
        best = self.path_length
        new_best = INF
        print best
        rand_choice = False
        while count < self.cycles and best > self.max_tour:
            if new_best == best:
                if rand_choice:
                    for n in self.G.nodes():
                        self._kopt(st = n, rand=True)
                    rand_choice = False
                else: 
                    for i in range(7300, self.N):
                        if i % 10 == 0:
                            print "Current node: {}".format(i)
                            if self.path_length < best:
                                best = self.path_length
                                print "saving..."
                                self.save()
                        self._kopt(st = self.G.node[i])
                    # rand_choice = True
            else:
                self._kopt(longest = True)
            new_best = self.path_length
            if new_best < best:
                print "New best: {}\nSaving...".format(new_best)
                self.save()
                best = new_best
            print "Count: {}".format(count)
            count += 1


    def _kopt(self, longest = False, st = None, rand=False):
        """swap k edges and choose best sequence of swaps"""
        ## Initialize vars
        added = {}
        removed = {}
        kvals = []
        k = 0
        pred = None
        k_init = self.path_length

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
            nebs = self._neighbors(self.points[n])
            if rand:
                nn = None
                while not nn:
                    nn = random.choice(nebs)[1]
                    if nn == n or n == pred: nn = None
            else:
                tries = 20
                curdist = self.G.edge[n].values().pop()['weight']
                accept = False
                while not accept and tries > 0:
                    nn = random.choice(nebs)[1]
                    if nn == n: 
                        accept = False
                    else:
                        ndist = length(self.points[nn], self.points[n])
                        accept = self._accept(curdist, ndist)
                        # print "ndist: {}, curdist: {}, accepted: {}"\
                            # .format(ndist, curdist, accept)
                    tries -= 1

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


    ####################################################################################
    ##                           helper functions
    def _accept(self, curdist, ndist, temp = None):
        """Decide whether to accept neighbor during k moves"""
        return float(curdist) / ndist > np.random.random()


    def _extract_dist(self, n1, n2):
        """Get the distance between two nodes"""
        return length(self.points[n1], self.points[n2])


    def _tour_length_path(self, path):
        """Calculate length of current tour"""
        tour_length = 0.0
        for i in range(self.N):
            if i > 0:
                p1 = self.points[path[i]]
                p2 = self.points[path[i-1]]
                tour_length += length(p1, p2)
        tour_length += length(self.points[path[self.N-1]],
                              self.points[path[0]])
        return tour_length


    ####################################################################################
    ##                           Graph management
    def _graphify(self):
        """Convert to graph data structure"""
        for i in range(1, len(self.path)):
            c1, c2 = self.path[i-1], self.path[i]
            self.G.add_edge(c1, c2, weight = length(self.points[c1], self.points[c2]))
        self.G.add_edge(self.path[i], self.path[0], weight =
                        length(self.points[self.path[i]], self.points[self.path[0]]))


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


    def _swap_edges(self, e1, e2):
        """Swap two edges in directed graph"""
        c1_t, c1_h, c2_t, c2_h = e1[0], e1[1], e2[0], e2[1]
        # Remove the two selected edges
        self.G.remove_edge(c1_t, c1_h)
        self.G.remove_edge(c2_t, c2_h)
        # Connect c1_t -> c2_t, c1_h -> c2_h
        self.G.add_edge(c1_t, c2_t, weight = length(self.points[c1_t], self.points[c2_t]))
        self.G.add_edge(c1_h, c2_h, weight = length(self.points[c1_h], self.points[c2_h]))


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


    ####################################################################################
    ##                              Neighborhood stuff
    def _make_gridlines(self):
        points = self.points
        dim = self.dim
        # find gridlines
        max_x = max([x for x,y in points])
        max_y = max([y for x,y in points])
        min_x = min([x for x,y in points])
        min_y = min([y for x,y in points])
        x_cuts = [min_x + 1.0/(dim) * i * (max_x - min_x) for i in range(dim+1)]
        y_cuts = [min_y + 1.0/(dim) * i * (max_y - min_y) for i in range(dim+1)]
        return x_cuts, y_cuts


    def _nhood(self, p):
        dim = self.dim
        for i in range(1,dim+1):
            for j in range(1,dim+1):
                if p[0] <= self.x_cuts[i]  and p[1] <= self.y_cuts[j]:
                    return (i-1, j-1)


    def _construct_hoods(self):
        """Initialize neighborhoods"""
        hoods = { (i, j) : [] for i in range(self.dim) for j in range(self.dim) }
        # (0,0) corresponds to x < x_cuts[1] and y < y_cuts[1]
        for i, p in enumerate(self.points):
            label = self._nhood(p)
            hoods[label].append((p, i))
        return hoods


    def _neighbors(self, p):
        """Returns neighbors of a points.  Neighborhoods default to 9 quadrats 
        (the 8 surrounding the 1 containing an individual), smaller for edge cases"""
        nsize = self.nsize
        q = self._nhood(p)
        dim = self.dim
        hood = []
        qs = [(q[0]-1+i, q[1]-1+j) for i in range(nsize) for j in range(nsize)]
        qs = [loc for loc in qs if loc[0] >= 0 and loc[0] <= dim-1\
              and loc[1] >= 0 and loc[1] <= dim-1]
        for h in qs:
            hood.extend(self.hoods[h])
        return hood
    

    def _cache_neighbors(self):
        """Cache neighbors for faster lookup"""
        dim = self.dim
        nsize = self.nsize
        neighborhoods = { (i, j) : [] for i in range(self.dim) for j in range(self.dim) }
        for q in self.hoods.keys():
            qs = [(q[0]-1+i, q[1]-1+j) for i in range(nsize) for j in range(nsize)]
            qs = [loc for loc in qs if loc[0] >= 0 and loc[0] <= dim-1\
                  and loc[1] >= 0 and loc[1] <= dim-1]
            for h in qs:
                neighborhoods[q].extend(self.hoods[h])
        self.hoods = neighborhoods


    ####################################################################################
    ##                           Setup -- initial solution
    def _initial_solution(self, st = None, rand = False):
        """Find initial solution, random configurations of points in each
        neighborhood joined together: [(0,0),(0,1),...,(n,0),(n,n)]"""
        dim = self.dim
        start = (0,0)
        if st:
            start = st
        elif rand:
            start = random.choice(range(self.N))
        # print "Starting from neighborhood {}".format(start)
        for x in range(6):
            if x % 2 == 0:
                for y in range(1, dim):
                    for n in self.hoods[(x, y)]:
                        self.path.append(n[1])
            else:
                for y in reversed(range(dim)):
                    for n in self.hoods[(x, y)]:
                        self.path.append(n[1])

        for y in range(1, dim):
            if y % 2 > 0:
                for x in range(6,35):
                    for n in self.hoods[(x, y)]:
                        self.path.append(n[1])
            else:
                for x in reversed(range(6, 35)):
                    for n in self.hoods[(x, y)]:
                        self.path.append(n[1])

        for x in range(35, dim):
            if x % 2 == 0:
                for y in range(1, dim):
                    for n in self.hoods[(x, y)]:
                        self.path.append(n[1])
            else:
                for y in reversed(range(dim)):
                    for n in self.hoods[(x, y)]:
                        self.path.append(n[1])

        for x in reversed(range(6, dim)):
            for n in self.hoods[(x, 0)]:
                self.path.append(n[1])


    ####################################################################################
    ##                                     I/O
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


    def save(self):
        """save path for visualization"""
        filename = '/home/noah/class/discopt/tsp/testing/solutions/' +\
                   'neighbor_anneal_' + str(self.N) + '.data'
        fp = open(filename, 'w')
        fp.write("%s " % self.path_length)
        for p in self.path:
            fp.write("%s " % p)
        fp.close()


    def _output(self):
        output_data = str(self.path_length) + ' ' + str(0) + '\n'
        output_data += ' '.join(map(str, self.path))
        return output_data

    ####################################################################################
    ##                      Distance matrix helpers          
    # def _extract_dist(self, i, j):
    #     """X[i, j] and X[j, i] values are set to 
    #     v[{n choose 2}-{n-i choose 2} + (j-i-1)]"""
    #     if i == j:
    #         return 0.0
    #     if i < j:
    #         i, j = j, i
    #     ind = (self.N * j) - (j * (j + 1) / 2) + (i - 1 - j)
    #     return self.dist[ind]

    # def _index_max(self, values):
    #     return max(xrange(len(values)), key = values.__getitem__)


    # def _index_min(self, values):
    #     return min(xrange(len(values)), key = values.__getitem__)


    # def _nearest_neighbor(self, i, seen = []):
    #     """Returns the nearest neighbor to point i"""
    #     dists = []
    #     for j in xrange(self.N):
    #         if j == i or j in seen:
    #             dists.append(INF)
    #         else:
    #             dists.append(self._extract_dist(i, j))
    #     # print dists
    #     return self._index_min(dists)
    

def solve_it(input_data):
    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])

    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    # p1: 430, p2: 20800, p3: 30000, p4: 37600, p5: 323000, p6: 78478868
    T = tsp(points)
    # T.save()
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

