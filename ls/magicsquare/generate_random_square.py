#!/usr/bin/python
import sys
import random


def generate_square(size):
    """Create square of consecutive numbers as test data for magic square"""
    nums = size * size
    seq = random.sample(list(range(1, nums + 1)), nums)
    output = ''
    for i in range(nums):
        if i > 0:
            output += " "
        if i > 0 and i % size == 0:
            output += '\n'
        output += "{0:3}".format(seq[i])
    return output

if __name__ == '__main__':
    if len(sys.argv) > 1:
        size = int(sys.argv[1].strip())
        print generate_square(size)
    else:
        print 'Usage: generate_random_square <int square size>'
