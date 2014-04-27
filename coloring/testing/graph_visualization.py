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
G = readData('../data/gc_4_1')
# Starting with most connected node, assign first color to as many
#  nodes as possible
n = len(G)
constraints = {i: list(reversed(range(n))) for i in G}
cols = {i: None for i in G}
uncolored = list(range(n))

# max connected node, take just the first if multiple
most = max([len(G.edge[i]) for i in G])
center = [c for c in G if len(G.edge[c]) == most].pop()
col = constraints[center].pop()
cols[center] = col

# color as many nodes same as center as possible
for nbr in G[center]:
    if col in constraints[nbr]:
        constraints[nbr].remove(col)

not_center = [n for n in G if center not in G.edge[n]]
temp = [uncolored.remove(_) for _ in not_center]
for n in not_center:
    constraints[n].pop()
    cols[n] = col
    for nbr in G[n]:
        if col in constraints[nbr]:
            constraints[nbr].remove(col)

# color nodes until all colored
stack = set(uncolored)
while(len(stack) > 0):
    temp = random.choice(list(stack))
    stack.remove(temp)
    assert cols[temp] == None, "Stack contains colored nodes"
    
    col = constraints[temp].pop()
    cols[temp] = col

    # color as many others as possible
    not_nbr = [neb for neb in stack if temp not in G.edge[neb]]
    for node in not_nbr:
        assert cols[node] == None, "Colored node found in stack"
        constraints[node].pop()
        cols[node] = col
        for nbr in G[node]:
            if col in constraints[nbr]:
                constraints[nbr].remove(col)
        stack.remove(node)

for comp in nx.connected_components(G):
    choice = random.choice(comp)
    stack = {choice}
    while (len(stack) > 0):
        temp = random.choice(list(stack))
        stack.remove(temp)

        if not constraints[temp]:
            break

        # Color choice
        col = constraints[temp].pop()
        cols[temp] = col

        # update neighbor constraints and add uncolored neighbors
        for nbr in G[temp]:
            assert col != cols[nbr], "Constraints failed at node %d" % nbr
            if col in constraints[nbr]:
                constraints[nbr].remove(col)
            if not cols[nbr]:
                stack.add(nbr)

cols.values()
