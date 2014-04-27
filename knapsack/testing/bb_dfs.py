#!/usr/bin/python
# Using branch and bound: first attempt
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import math
from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])

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

# test data
# n, W, items = readData('../data/ks_50_0')

def solve_it(input_data):
    class knapsack:
        """Fills a knapsack with capacity W, choosing from 
        n items each with a weight and value.
        Chooses optimal solution by branch and bound method
        with depth-first search"""

        def __init__(self, items, W):
            self.n = len(items)
            dtype = [('index', int), ('value', int), ('weight', int), ('ratio', float)]
            values = []
            for i in range(len(items)):
                values.append((i, items[i][0], items[i][1], items[i][0]/items[i][1]))
            items = np.sort(np.array(values, dtype = dtype), order = 'ratio')
            self.items = np.array(list(reversed(items)))
            self.W = W
            self.taken = [0] * self.n
            self.estimate = self._estimate(0, self.W, 0)
            self.maxvalue = sum([j for i,j,k,l in self.items])
            # fill the knapsack
            self._stuffsack()
            self.value = self._optimum()

        def __len__(self):
            return self.n

        def __getitem__(self, i):
            return list(self.taken)[i]

        def _fixcontents(self, taken):
            """Fix the indexes of the taken items"""
            new_taken = [0] * n
            real_inds = [j for i,(j,k,l,m) in zip(taken, self.items) if i > 0]
            for i in real_inds:
                new_taken[i] = 1
            return new_taken

        def __str__(self):
            """First line: optimal value knapsack, flag whether optimal
            Second line: {0,1} indices of items included in optimal knapsack"""
            output_data = str(self.value) + ' ' + str(0) + '\n'
            output_data += ' '.join(map(str, self.taken))
            return output_data

        def output(self):
            """Return output string"""
            output_data = str(self.value) + ' ' + str(0) + '\n'
            output_data += ' '.join(map(str, self.taken))
            return output_data

        def _estimate(self, j, w, val):
            """Get estimate for maximum value: value/weight linear relaxation"""
            est = val; space = w;
            for i in self.items[j:]:
                if i[2] > space:
                    est += (space/i[2]) * i[1]
                    break
                else:
                    est += i[1]
                    space -= i[2]
            return int(math.floor(est))

        def _optimum(self):
            return sum([i * k for i, (j,k,l,m) in zip(self.taken, self.items)])

        def _stuffsack(self):
            """Stuff the sack"""
            take = [0] * self.n
            take[0] = 1
            opt = self._branch(0, 0, 0, self.estimate, 0, take)
            take[0] = 0
            opt = self._branch(0, 0, 0, self.estimate, opt, take)

        def _branch(self, i, val, wt, est, opt, take):
            """Branch by depth first search"""
            weight = wt + self.items[i][2] * take[i]
            if weight > self.W:
                return opt
            value = val + self.items[i][1] * take[i]
            cur_est = self._estimate(i, self.W - weight, value)
            # print cur_est
            # print opt
            if cur_est < opt:
                return opt
            new_opt = max(value, opt)
            # if not last item, recurse both branches, returning optimal
            if i < self.n - 1:
                take[i+1] = 1
                new_opt = max(self._branch(i+1, value, weight, cur_est, new_opt, take), new_opt)
                take[i+1] = 0
                new_opt = max(self._branch(i+1, value, weight, cur_est, new_opt, take), new_opt)
                # take[i+1] = None
            elif value >= opt:
                self.taken = self._fixcontents(list(take))
            return new_opt

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    n = int(firstLine[0])
    W = int(firstLine[1])
    
    # rework input to be list of tuples (value, weight)
    things1 = [l.split() for l in lines[1:] if l]
    items = np.array([(int(l[0]), int(l[1])) for l in things1])

    # Find best solution
    k = knapsack(items, W)

    # prepare the solution in the specified output format
    return k.output()


import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        input_data_file = open(file_location, 'r')
        input_data = ''.join(input_data_file.readlines())
        input_data_file.close()
        print solve_it(input_data)
    else:
        print 'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)'
