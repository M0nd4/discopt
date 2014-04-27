from collections import namedtuple

Point = namedtuple("Point", ['x', 'y'])

file_location = '../data/tsp_574_1'
input_data_file = open(file_location, 'r')
input_data = ''.join(input_data_file.readlines())
input_data_file.close()

lines = input_data.split('\n')

nodeCount = int(lines[0])

points = []
for i in range(1, nodeCount+1):
    line = lines[i]
    parts = line.split()
    points.append(Point(float(parts[0]), float(parts[1])))

# # Function to read data to directed graph
# def graphify(G):
#     """Convert points to directed graph data structure"""
#     for e in range(len(G.edges)):
#         c1, c2 = self._cities_from_edge(e)
#         G.add_edge(c1, c2, weight = self.edges[e])

# file_location = './solutions/threeOpt_random_574.data'
# input_data_file = open(file_location, 'r')
# input_data = ''.join(input_data_file.readlines())
# input_data_file.close()

# lines = input_data.split()
# pp = map(int, lines[1:])
