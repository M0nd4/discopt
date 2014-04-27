#!/usr/bin/python
# Using dynamic programming approach, keep only two columns to reduce memory
# -*- coding: utf-8 -*-
# import numpy as np
from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])


def solve_it(input_data):
    # Process two columns at a time, calculate best set of items at each step
    def knapsack(n, W, items):
        best = ['0'*n for _ in range(W+1)]
        K = [x[:] for x in [[0]*(W+1)]*2]
        current = 1
        while current <= n:
            K = buildcol(current, W, K, items)
            best = traceback(current, W, K, items, best)
            if current < n:
                K = rearrange(W, K)
            current += 1
        return K[1][W], best[W]

    def buildcol(current, W, K, items):
        """Adds a single column"""
        i = current
        for j in range(W+1):
            if j == 0:
                K[1][j] = 0
            elif items[i-1][1] <= j:
                K[1][j] = max([items[i-1][0] + K[0][j-items[i-1][1]], 
                               K[0][j]])
            else:
                K[1][j] = K[0][j]
        return K

    def rearrange(W, K):
        """rearrange K for next iteration"""
        newK = [x[:] for x in [[0]*(W+1)]*2]
        newK[0] = K[1]
        return newK

    def traceback(current, W, K, items, best):
        """Update best solutions from the last increment"""
        wt = items[current-1][1];
        for j in range(W + 1):
            if (K[1][j] != K[0][j]):
                newBest = list(best[j-wt])
                newBest[current-1] = '1'
                best[j] = ''.join(newBest)
        return best

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    n = int(firstLine[0])
    W = int(firstLine[1])
    
    # rework input to be list of tuples (value, weight)
    things1 = [l.split() for l in lines[1:] if l]
    items = [(int(l[0]), int(l[1])) for l in things1]

    # Find best solution
    value, taken = knapsack(n, W, items)
    
    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(1) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data


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


# def readData(file_location):
#     input_data_file = open(file_location, 'r')
#     input_data = ''.join(input_data_file.readlines())
#     input_data_file.close()
#     return input_data
