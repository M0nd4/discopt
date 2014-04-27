#!/usr/bin/python
# -*- coding: utf-8 -*-
import networkx as nx
import matplotlib.pyplot as plt
from k_opt import tsp
import numpy as np

from collections import namedtuple

Point    = namedtuple("Point", ['x', 'y'])

# fp = open('../data/tsp_33810_1', 'r')
fp = open('../data/tsp_51_1', 'r')
# fp = open('../data/tsp_100_3', 'r')
# fp = open('../data/tsp_574_1', 'r')

data = ''.join(fp.readlines())
fp.close()

lines = data.split('\n')
ps = [map(float, l.split()) for l in lines[1:] if l]
points = [Point(x, y) for x,y in ps]

# T = tsp(points)
pos = {i:points[i] for i in range(len(points))}

## Create neighborhoods with location gridding
# sort by x-coord and distance from (0,0)
tst = sorted(points, key = lambda x: (x[0], np.sqrt((x[0]**2 + x[1]**2))))

# Decide how many grids to make
max_x = max([x for x,y in points])
max_y = max([y for x,y in points])
min_x = min([x for x,y in points])
min_y = min([y for x,y in points])

dim = 40
x_cuts = [1.0/(dim) * i * (max_x - min_x) for i in range(dim+1)]
y_cuts = [1.0/(dim) * i * (max_y - min_y) for i in range(dim+1)]

# start with 100 groups

def nhood(p):
    for i in range(1,dim+1):
        for j in range(1,dim+1):
            if p[0] <= x_cuts[i]  and p[1] <= y_cuts[j]:
                return (i-1, j-1)

# (0,0) corresponds to x < x_cuts[1] and y < y_cuts[1]
hoods = { (i, j) : [] for i in range(dim) for j in range(dim) }
for p in points:
    label = nhood(p)
    hoods[label].append(p)

pops = [len(hoods[i]) for i in hoods]

# function to return neighbors of point
def neighbors(q, nsize = 3):
    """Neighborhoods default to 9 quadrats (the 8 surrounding the 1 containing
    an individual), smaller for edge cases"""
    hood = []
    qs = [(q[0]-1+i, q[1]-1+j) for i in range(nsize) for j in range(nsize)]
    qs = [loc for loc in qs if loc[0] >= 0 and loc[0] <= dim-1\
          and loc[1] >= 0 and loc[1] <= dim-1]
    for h in qs:
        hood.extend(hoods[h])
    return hood


## inital path to connect
dim = 40
for x in range(6):
    # vertical pathing for the first and last columns
    if x % 2 == 0:
        for y in range(1, dim):
            print "({}, {})".format(x, y)
    else:
        for y in reversed(range(dim)):
            print "({}, {})".format(x, y)

for y in range(1, dim):
    if y % 2 > 0:
        for x in range(6,35):
            print "({}, {})".format(x, y)
    else:
        for x in reversed(range(6, 34)):
            print "({}, {})".format(x, y)

for x in range(35, dim):
    if x % 2 == 0:
        for y in range(1, dim):
            print "({}, {})".format(x, y)
    else:
        for y in reversed(range(dim)):
            print "({}, {})".format(x, y)

for x in reversed(range(6, dim)):
    print "({}, {})".format(x, 0)
