#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
# from itertools import combinations
import sys
import numpy as np
import math

# def readData(file_location):
#     input_data_file = open(file_location, 'r')
#     input_data = ''.join(input_data_file.readlines())
#     input_data_file.close()
#     # parse the input
#     lines = input_data.split('\n')

#     firstLine = lines[0].split()
#     n = int(firstLine[0])
#     W = int(firstLine[1])
    
#     # rework input to be list of tuples (value, weight)
#     things1 = [l.split() for l in lines[1:] if l]
#     items = np.array([(int(l[0]), int(l[1])) for l in things1])

#     return n, W, items

# # test data
# n, W, items = readData('../data/ks_4_0')

def solve_it(input_data):
    class Node:
        def __init__(self, level, value, weight, members):
            self.level = level; self.weight = weight; self.value = value
            self.members = members

        def add(self, ind):
            self.members.append(ind)

        def __str__(self):
            return str((self.level, self.value, self.weight, self.members))

    class knapsack:
        """Fills a knapsack with capacity K, choosing from 
        n items each with a weight and value.
        Chooses optimal solution by branch and bound method
        with depth  first search (no recursion - use stack instead for large inputs)
        Usage: ./<solver-name> <data file>
        """

        def __init__(self, K, values, weights):
            assert len(values) == len(weights), "wrong input dimensions"
            self.n     = len(values)
            self.items = sorted( [(i, (v, w)) for i, (v, w)
                                  in enumerate(zip(values, weights))],
                                 key = lambda y: y[1][0] / y[1][1], reverse = True )
            self.wts   = [w for i, (v, w) in self.items]
            self.vals  = [v for i, (v, w) in self.items]
            self.K     = K

            # fill the knapsack
            self.best = self._greedy()
            self._solve()
            self.taken = self._fixcontents(self.best.members)
            self.value = self.best.value

        def _greedy(self):
            w = v = 0
            t = []
            for i, (val, wt) in self.items:
                if w + wt <= self.K:
                    v += val
                    w += wt
                    t.append(i)
            return Node(i, v, w, t)

        def _bound(self, i, value, weight):
            """Determine upper bound by linear relaxation: value/weight"""
            bound = value
            total_weight = weight
            k = i + 1
            if weight > self.K:
                return 0
            while (k < self.n) and (total_weight + self.wts[k] <= self.K):
                bound += self.vals[k]
                total_weight += self.wts[k]
                k += 1
            if k < self.n:
                bound += (self.K - total_weight) * (self.vals[k] / self.wts[k])
            return bound

        def _stuffit(self, root):
            """Fill the knapsack"""
            best = self.best
            stack = [root]
            while (len(stack) > 0):
                temp = stack.pop()
                i = temp.level
                if temp.value >= best.value:
                    best = temp
                
                if i < self.n:
                    # make node that contains next item if possible
                    if temp.weight + self.wts[i] <= self.K:
                        eaten = Node(i + 1, temp.value + self.vals[i], 
                                     temp.weight + self.wts[i], list(temp.members))
                        eaten.add(i)
                        if eaten.value > best.value:
                            best = eaten
                        bound = self._bound(i, eaten.value, eaten.weight)
                        if bound >= best.value:
                            stack.append(eaten)

                    # Now make node without next item
                    starved = Node(i + 1, temp.value, temp.weight, list(temp.members))
                    bound = self._bound(i, starved.value, starved.weight)
                    if bound >= best.value:
                        stack.append(starved)
            self.best = best

        def _solve(self):
            root = Node(0, 0, 0, [])
            self._stuffit(root)

        def __len__(self):
            return self.n

        def __getitem__(self, i):
            return list(self.taken)[i]

        def _fixcontents(self, inds):
            """Fix the indexes of the taken items to be {0, 1} in 
            original usorted order"""
            new_taken = [0] * self.n
            items = [self.items[i] for i in inds]
            original_inds = [i for i, (v,w) in items]
            for i in original_inds:
                new_taken[i] = 1
            return new_taken

        def __str__(self):
            """First line: optimal value knapsack, flag whether optimal
            Second line: {0,1} indices of items included in optimal knapsack"""
            output_data = str(self.value) + ' ' + str(1) + '\n'
            output_data += ' '.join(map(str, self.taken))
            return output_data

        def output(self):
            """Return output string"""
            output_data = str(self.value) + ' ' + str(1) + '\n'
            output_data += ' '.join(map(str, self.taken))
            return output_data

        def _optimum(self):
            dtype = [('index', int), ('value', int), ('weight', int), ('ratio', float)]
            items = np.sort(np.array(self.items, dtype=dtype), order = 'index')
            return sum([i * k for i, (j,k,l,m) in zip(self.taken, items)])

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    n = int(firstLine[0])
    W = int(firstLine[1])
    
    # rework input to be list of tuples (value, weight)
    things1 = [l.split() for l in lines[1:] if l]
    items = [(int(l[0]), int(l[1])) for l in things1]
    vals = [v for v,w in items]
    wts = [w for v,w in items]

    # Find best solution
    k = knapsack(W, vals, wts)

    # prepare the solution in the specified output format
    return k.output()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        input_data_file = open(file_location, 'r')
        input_data = ''.join(input_data_file.readlines())
        input_data_file.close()
        print solve_it(input_data)
    else:
        print 'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)'
