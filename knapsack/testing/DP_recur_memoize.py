#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])


def solve_it(input_data):
    # Modify this code to run your optimization algorithm
    ## Recursive version to handle larger memory case, will store each recursive call in
    ## python dictionary for lookup later
    def recursiveKnapsack(items, maxweight):
        P = {}
        indices = [0]*len(items)
        def loop(numItems, lim):
            if (numItems, lim) not in P:
                if numItems == 0:
                    P[(numItems, lim)] = 0
                elif items[numItems-1].weight > lim:
                    P[(numItems, lim)] = loop(numItems-1,lim)
                else:
                    opt1 = loop(numItems-1, lim)
                    opt2 = loop(numItems-1, lim - items[numItems-1][2]) + items[numItems-1][1]
                    if opt2 > opt1:
                        indices[items[numItems-1].index] = 1
                    else:
                        indices[items[numItems-1].index] = 0
                    P[(numItems, lim)] = max(opt1, opt2)
            return P[(numItems, lim)]
        return [loop(len(items), maxweight), indices, P]

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    # rework input to be list of tuples (value, weight)
    # things1 = [l.split() for l in lines[1:] if l]
    # things = [(int(l[0]), int(l[1])) for l in things1]
    sys.setrecursionlimit(6000)

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))

    value, taken, P = recursiveKnapsack(items, capacity)
    # print val

    # a trivial greedy algorithm for filling the knapsack
    # it takes items in-order until the knapsack is full
    # value = 0
    # weight = 0
    # taken = [0]*len(items)

    # for item in items:
    #     if weight + item.weight <= capacity:
    #         taken[item.index] = 1
    #         value += item.value
    #         weight += item.weight
    
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
