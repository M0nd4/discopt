#!/usr/bin/python
# local search to find magic square
# (sums of rows, cols, and diagonals all the same)
import sys
import random
from copy import deepcopy

class magicSquare:
    """Takes a normal square (i.e n * n, n in range(1, n+1)) and
    creates the magic square (i.e sum(rows) == sum(cols) == sum(diagonals))"""
    def __init__(self, matrix):
        self.initial_matrix = deepcopy(matrix)
        self.matrix = matrix
        self.dim = len(self.matrix[0])
        self.m = self._determine_constant()
        self.constraints = self._calculate_constraints()
        self._create_magic_square()


    def _determine_constant(self):
        """Determine the constant of the *normal* magic square"""
        n = self.dim
        m = float((n * (n * n + 1))) / 2
        return m

    def _calculate_constraints(self):
        """Calculate the current constraints"""
        matrix = self.matrix
        m = self.m
        dim = self.dim
        constraints = []
        for row in matrix:
            constraints.append(m - sum(row))
        for col in range(dim):
            constraints.append(m - sum([matrix[i][col]
                                        for i in range(dim)]))
        constraints.append(m - sum([matrix[i][i] for i in range(dim)]))
        constraints.append(m - sum([matrix[i][dim - 1 - i]
                                    for i in range(dim)]))
        return map(abs, constraints)

    def _swap_numbers_randomly(self):
        """swap two numbers in square"""
        matrix = self.matrix
        inds = list(range(self.dim))
        x1 = random.choice(inds)
        x2 = random.choice(inds)
        y1 = random.choice(inds)
        y2 = random.choice(inds)
        matrix[x1][y1], matrix[x2][y2] = matrix[x2][y2], matrix[x1][y1]

    def _swap_inds(self, ind1, ind2):
        """swap two indices in matrix"""
        matrix = self.matrix
        dim = self.dim
        ind1 = self._convert_ind(ind1)
        ind2 = self._convert_ind(ind2)
        matrix[ind2[0]][ind2[1]], matrix[ind1[0]][ind1[1]]\
            = matrix[ind1[0]][ind1[1]], matrix[ind2[0]][ind2[1]]

    def _convert_ind(self, ind):
        """convert numerical index to tuple"""
        dim = self.dim
        index = (ind % dim, ind / dim)
        return index

    def _swap_numbers_constrained(self):
        """Swap numbers so that constraints always decrease"""
        consts = self.constraints
        dim = self.dim
        inds = list(range(dim*dim))
        matrix = self.matrix
        new_consts = list(consts)
        assert  sum(new_consts) > 0, "Looping after problem solved"
        for i in inds:
            for j in inds[i:]:
                if i != j:
                    self._swap_inds(i, j)
                    temp_consts = self._calculate_constraints()
                    if sum(temp_consts) < sum(self.constraints):
                        # print "new constraints: {}, old: {}".format(sum(temp_consts), 
                        #                                             sum(self.constraints))
                        self.constraints = list(temp_consts)
                        return
                    else:
                        self._swap_inds(i, j)

        # If stuck in a configuration with no good move, choose randomly
        self._swap_numbers_randomly()
        self.constraints = self._calculate_constraints()

    def _create_magic_square(self):
        """Convert square of numbers to magic square"""
        while (sum(self.constraints) > 0):
            self._swap_numbers_constrained()

    def _generate_output(self):
        """Generate ouput string"""
        matrix = self.matrix
        output = ''
        for row in matrix:
            for i in range(len(row)):
                if i > 0:
                    output += " "
                output += "{0:3}".format(row[i])
            output += '\n'
        return output

    def __str__(self):
        return self._generate_output()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        input_file = sys.argv[1].strip()
        input_data = open(input_file, 'r')
        data = ''.join(input_data.readlines())
        input_data.close()
        # convert to matrix
        lines = data.split('\n')
        matrix = [[int(i) for i in row.split()] for row in lines if row]
        ms = magicSquare(matrix)
        print ms
    else:
        print 'Usage: create_random_square <file>'
