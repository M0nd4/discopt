#!/usr/bin/python
# -*- coding: utf-8 -*-
import networkx as nx
import random

def solve_it(input_data):
    def color_graph(G, max_colors):
        """Assign as many nodes as possible with one color, then do local 
        search to find violations and relace as necessary"""
        # Starting with most connected node, assign first color to as many
        #  nodes as possible
        n = len(G)
        constraints = {i: list(reversed(range(n))) for i in G}
        cols = {i: None for i in G}
        uncolored = set(G.node)

        # max connected node, take just the first if multiple
        most = max([len(G.edge[i]) for i in G])
        high_traffic_nodes = [c for c in G if len(G.edge[c]) >= most - 3]
        center = random.choice(high_traffic_nodes)
        col = constraints[center].pop()
        cols[center] = col
        uncolored.remove(center)

        # color as many nodes same as center as possible
        for nbr in G[center]:
            constraints[nbr].remove(col)

        not_center = [n for n in G if center not in G.edge[n]]
        for node in not_center:
            if col in constraints[node]:
                constraints[node].remove(col)
                cols[node] = col
                uncolored.remove(node)
                for nbr in G[node]:
                    if col in constraints[nbr]:
                        constraints[nbr].remove(col)
        # Check constraints
        # for node in G:
        #     for nbr in G[node]:
        #         assert cols[node] not in constraints[nbr], "checkpoint 1..."

        # color nodes until all colored
        # uncolored = [i for i in cols if cols[i] == None]
        # assert len(uncolored) == len([i for i in cols if cols[i] == None]), "uncolored wrong"
        stack = set(uncolored)
        # assert all([cols[i] == None for i in stack]), "stack colored before loop"
        count = 0
        while(len(stack) > 0):
            temp = random.choice(list(stack))
            stack.remove(temp)
            # assert cols[temp] == None, "Stack contains colored nodes"
            # Check constraints
            # for node in G:
            #     for nbr in G[node]:
            #         assert cols[node] not in constraints[nbr], "checkpoint 2... count: %d"\
            #             % (count)

            col = constraints[temp].pop()
            cols[temp] = col
            if col > max_colors:
                return float('Inf'), []

            # update neighbor constraints
            for nbr in G[temp]:
                if col in constraints[nbr]:
                    constraints[nbr].remove(col)

            # color as many others as possible
            not_nbr = [neb for neb in stack if temp not in G.edge[neb]]
            for node in not_nbr:
                # assert cols[node] == None, "Colored node found in stack"
                if col in constraints[node]:
                    constraints[node].remove(col)
                    cols[node] = col
                    for nbr in G[node]:
                        if col in constraints[nbr]:
                            constraints[nbr].remove(col)
                    stack.remove(node)
            count += 1
        # Check constraints
        # for node in G:
        #     for nbr in G[node]:
        #         assert cols[node] not in constraints[nbr], "Final check..."

        return len(set(cols.values())), cols.values()

    # parse the input
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

    # build a solution
    solution = []
    max_colors = 17
    while not solution or count > max_colors:
        count, solution = color_graph(G, max_colors)

    # prepare the solution in the specified output format
    output_data = str(count) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

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
        print 'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)'

