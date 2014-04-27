#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
from subprocess import Popen, PIPE

def solve_it(input_data):
    # def readData(file_location):
    #     input_data_file = open(file_location, 'r')
    #     input_data = ''.join(input_data_file.readlines())
    #     input_data_file.close()
    #     return input_data

    # Write the inputData to a temporay file
    tmp_file_name = 'tmp.data'
    tmp_file = open(tmp_file_name, 'w')
    tmp_file.write(input_data)
    tmp_file.close()

    # Run the command:
    process = Popen(['./knapsack' , tmp_file_name], stdout=PIPE)
    (stdout, stderr) = process.communicate()

    # remove the temporay file
    os.remove(tmp_file_name)

    # stupid workaround
    # temp = open('temp.out', 'w')
    # temp.write(stdout.strip())
    # res = readData('temp.out')
    data = stdout.strip()

    # parse the input
    lines = data.split('\n')

    firstLine = lines[0].split()
    value = int(firstLine[0])
    res = lines[1].split()
    taken = []
    for i in range(len(res)):
        taken.append(int(res[i]))
    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
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
        print 'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)'

