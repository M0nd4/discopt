#!/usr/bin/python
# Using branch and bound: first attempt
# -*- coding: utf-8 -*-
from __future__ import division
from itertools import combinations
import sys
import numpy as np
import math
import Queue

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
        def __init__(self):
            self.level = 0
            self.size = self.value = self.bound = 0.0
            self.contains = []
            
        def add(self, ind):
            self.contains.append(ind)

    class knapsack:
        """Fills a knapsack with capacity K, choosing from 
        n items each with a weight and value.
        Chooses optimal solution by branch and bound method
        with breadth first search (no recursion, but is very slow and uses much memory)
        Usage: items is array of (value, weight) of items, K is knapsack capacity
        """

        def __init__(self, K, items):
            self.n = len(items)
            dtype = [('index', int), ('value', int), ('weight', int), ('ratio', float)]
            values = []
            for i in range(len(items)):
                values.append((i, items[i][0], items[i][1], items[i][0]/items[i][1]))
            items = np.sort(np.array(values, dtype = dtype), order = 'ratio')
            self.items = np.array(list(reversed(items)))
            self.s = [w for i,v,w,r in items]
            self.v = [v for i,v,w,r in items]
            self.inds = [i for i,v,w,r in items]
            self.K = W
            self.taken = []
            self.max_value = 0.0
            self.q = Queue.Queue()

            # fill the knapsack
            self._solve()
            self.taken = self._fixcontents()
            self.value = int(self.max_value)

        def _stuff(self):
            while (not self.q.empty()):
                temp = self.q.get()
                if temp.bound > self.max_value:
                    u = Node()
                    u.level = temp.level + 1
                    u.size = temp.size + self.s[temp.level + 1]
                    u.value = temp.value + self.v[temp.level + 1]
                    u.contains = list(temp.contains)
                    u.add(temp.level + 1)
                    if (u.size <= self.K) and (u.value >= self.max_value):
                        self.max_value = u.value
                        self.taken = list(u.contains)
                    u.bound = self._bound(u.level, u.size, u.value)
                    if u.bound > self.max_value:
                        self.q.put(u)
                    w = Node()
                    w.level = temp.level + 1
                    w.size = temp.size
                    w.value = temp.value
                    w.contains = list(temp.contains)
                    # w.add(temp.level)
                    w.bound = self._bound(w.level, w.size, w.value)
                    if w.bound >= self.max_value:
                        self.q.put(w)

        def _bound(self, item, size, value):
            bound = value
            total_size = size
            k = item + 1
            if size > self.K:
                return 0.0
            while (k < self.n) and (total_size + self.s[k] <= self.K):
                bound += self.v[k]
                total_size += self.s[k]
                k += 1
            if k < self.n:
                bound += (self.K - total_size) * (self.v[k] / self.s[k])
            return bound
        
        def _solve(self):
            root = Node()
            root.bound = self._bound(0, 0.0, 0.0)
            self.q.put(root)
            self._stuff()

        def __len__(self):
            return self.n

        def __getitem__(self, i):
            return list(self.taken)[i]

        def _fixcontents(self):
            """Fix the indexes of the taken items to be {0, 1} in 
            original usorted order"""
            new_taken = [0] * self.n
            inds = [self.inds[i] for i in self.taken]
            for i in inds:
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
    items = np.array([(int(l[0]), int(l[1])) for l in things1])

    # Find best solution
    k = knapsack(W, items)

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
