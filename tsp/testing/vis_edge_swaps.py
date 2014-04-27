import os
os.chdir('/home/noah/class/discopt/tsp/testing')

import matplotlib.pyplot as plt
import networkx as nx

# sample data from tsp_51_1
from loadData import *
from threeOpt import tsp

# Make directed graph
T = tsp(points)
G = T.G

nx.draw(G)
plt.show()

# color one edge
ecols = ['black'] * len(G.edges())
ecols[10] = 'r'
nx.draw_networkx(G, edge_color = ecols)
