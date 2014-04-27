#!/usr/bin/python
# -*- coding: utf-8 -*-
import networkx as nx
import random

def solve_it(input_data):
    def color_graph(G, max_colors):
        """Color successive maximal subsets until all vertices are colored,
        simple program that returns decent but not optimal results"""
        subgraph = G.copy()
        n = len(G)
        constraints = {i: list(reversed(range(n))) for i in G}
        cols = {i: None for i in G}
        num_colored = len([i for i in cols if cols[i] != None])
        # print str(len(G))

        count = 0
        while(num_colored < n):
            # color the maximal set
            ind_set = nx.maximal_independent_set(subgraph)
            if len(ind_set) == 0:
                return 0, []
            for node in ind_set:
                col = constraints[node].pop()
                cols[node] = col
                num_colored += 1
                subgraph.remove_node(node)
                for nbr in G[node]:
                    if col in constraints[nbr]:
                        constraints[nbr].remove(col)

            # print "ind set length: %d, col %d" % (len(ind_set), col)
            
            # check constraints
            # for node in G:
            #     for nbr in G[node]:
            #         assert cols[node] not in constraints[nbr],\
            #             "Constraints violated after %d iteration" % (count)
            #         if cols[node]:
            #             assert all([cols[node] != cols[nbr]]),\
            #                 "Constraints violated after %d iteration, (%d, %d)"\
            #                 % (count, node, nbr)
            count += 1

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
    max_colors = 1000
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

