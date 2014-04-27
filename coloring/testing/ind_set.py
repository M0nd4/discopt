import networkx as nx
import matplotlib.pyplot as plt
import random

# read data
def readData(file_location):
    """Reads in graph data"""
    input_data_file = open(file_location, 'r')
    input_data = ''.join(input_data_file.readlines())
    input_data_file.close()

    lines = input_data.split('\n')
    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    # Create graph
    G = nx.Graph()
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        G.add_edge(int(parts[0]), int(parts[1]))
    return G

nx.draw(G)

plt.show()
nx.connected_components(G)
nx.isolates(G)

# Data
G = readData('../data/gc_20_1')

# successively color maximal subsets until all vertices are colored
subgraph = G.copy()
n = len(G)
constraints = {i: list(reversed(range(n))) for i in G}
cols = {i: None for i in G}
num_colored = len([i for i in cols if cols[i] != None])

count = 0
while(num_colored < n):
    # color the maximal set
    ind_set = nx.maximal_independent_set(subgraph)
    for node in ind_set:
        col = constraints[node].pop()
        cols[node] = col
        for nbr in G[node]:
            if col in constraints[nbr]:
                constraints[nbr].remove(col)

    # Now do greedy coloring
    # subgraph = G.copy()
    colored = [i for i in cols if cols[i] != None]
    num_colored = len(colored)
    for node in colored:
        if node in subgraph:
            subgraph.remove_node(node)

    # check constraints
    for node in G:
        for nbr in G[node]:
            assert cols[node] not in constraints[nbr],\
                "Constraints violated after %d iteration" % (count)
            if cols[node]:
                assert all([cols[node] != cols[nbr]]),\
                    "Constraints violated after %d iteration, (%d, %d)" % (count, node, nbr)
    count += 1

