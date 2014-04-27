#!/usr/bin/python
# -*- coding: utf-8 -*-
import networkx as nx
import matplotlib.pyplot as plt
from k_opt import tsp

from collections import namedtuple

Point    = namedtuple("Point", ['x', 'y'])

fp = open('testGraph.txt', 'r')
data = ''.join(fp.readlines())
fp.close()

lines = data.split('\n')
ps = [map(float, l.split()) for l in lines[1:] if l]
points = [Point(x, y) for x,y in ps]

T = tsp(points)
pos = {i:points[i] for i in range(len(points))}
## walkthrough kopt

