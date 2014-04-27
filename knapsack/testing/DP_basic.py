#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])


def solve_it(input_data):

    # Simple bottom up knapsack
    def knapsack(n, W, items):
        K = [x[:] for x in [[0]*(W+1)]*(n+1)]
        for i in range(n+1):
            for j in range(W+1):
                if i == 0 or j == 0:
                    K[i][j] = 0
                elif items[i-1][1] <= j:
                    K[i][j] = max([items[i-1][0] + K[i-1][j-items[i-1][1]], K[i-1][j]])
                else:
                    K[i][j] = K[i-1][j]
        return K

    # get the items taken with traceback
    def traceback(n, W, K, items):
        i = n; j = W;
        taken = [0]*n
        while(i > 0):
            if (K[i][j] != K[i-1][j]):
                taken[i-1] = 1
                j -= items[i-1][1]
            i -= 1
        return taken

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    n = int(firstLine[0])
    W = int(firstLine[1])
    
    # rework input to be list of tuples (value, weight)
    things1 = [l.split() for l in lines[1:] if l]
    items = [(int(l[0]), int(l[1])) for l in things1]

    K = knapsack(n, W, items)
    value = K[n][W]
    taken = traceback(n, W, K, items)
    
    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
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
