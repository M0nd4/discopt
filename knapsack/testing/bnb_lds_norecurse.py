#!/usr/bin/python
# Using branch and bound: first attempt
# -*- coding: utf-8 -*-
from __future__ import division
from itertools import combinations
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
# n, W, items = readData('./data/ks_4_0')

def solve_it(input_data):
    class knapsack:
        """Fills a knapsack with capacity W, choosing from 
        n items each with a weight and value.
        Chooses optimal solution by branch and bound method
        with local discrepancy search (no recursion - for large problems where recursion
        depth is an issue)
        Usage: items is array of (value, weight) of items, W is knapsack capacity
        """

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
            dtype = [('index', int), ('value', int), ('weight', int), ('ratio', float)]
            items = np.sort(np.array(self.items, dtype=dtype), order = 'index')
            return sum([i * k for i, (j,k,l,m) in zip(self.taken, items)])

        def _wts_vals(self, take):
            """return weight and value at each node on optimal tree"""
            new_wts = []
            new_vals = []
            temp = zip(take, self.items)
            for i in range(1, self.n+1):
                # v_w = [(t*v, t*w) for t,(ind,v,w,r) in temp[i:]]
                new_wts.append(sum([t*w for t,(ind,v,w,r) in temp[:i]]))
                new_vals.append(sum([t*v for t,(ind,v,w,r) in temp[:i]]))
            return new_wts, new_vals

        def _update_wt_val(self, i, wts, vals, new_wt, new_val):
            """Update node wts and values on optimal tree with new value and weight"""
            wts[i] = new_wt
            vals[i] = new_val
            if i < self.n - 1:
                for j in range(i+1, self.n):
                    wts[j] = new_wt
                    vals[j] = new_val
            return wts, vals

        def _stuffsack(self):
            """Stuff the sack"""
            choices = opt = 0
            # Keep track of the best path
            take = [0] * self.n
            wts = [0] * self.n
            vals = [0] * self.n

            for choice in range(self.n):
                # print choice

                for i in range(self.n):
                    cs = choice
                    if i > 0:
                        new_wt = wts[i-1]
                        new_val = vals[i-1]
                    else:
                        new_wt = new_val = 0
                    path = list(take)

                    for j in range(i, self.n):
                        t = path[j]
                        if t == 1 and cs > 0: # remove the item
                            t = 0
                            cs -= 1
                        else:
                            t = 1
                        new_wt += self.items[j][2] * t
                        if new_wt > self.W:
                            break
                        new_val += self.items[j][1] * t
                        path[j] = t
                        if new_val >= opt:
                            take = list(path)
                            opt = new_val
                            wts, vals = self._wts_vals(take)
                        cur_est = self._estimate(j, self.W - new_wt, new_val)
                        if cur_est < opt:
                            break
                    if new_val >= opt and new_wt <= self.W and i == self.n - 1:
                        take = list(path)
                        wts, vals = self._wts_vals(take)
                        # print str(take) + ', ' + str(new_wt) + \
                        #     ', ' + str(new_val) + ', ' + str(cur_est) + ', ' + str(opt)
            self.taken = self._fixcontents(take)

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

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        input_data_file = open(file_location, 'r')
        input_data = ''.join(input_data_file.readlines())
        input_data_file.close()
        print solve_it(input_data)
    else:
        print 'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)'
