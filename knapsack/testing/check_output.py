solver = __import__('solver-dp-trim-numpy')
import os
import sys
import numpy as np

def readData(file_location):
    input_data_file = open(file_location, 'r')
    input_data = ''.join(input_data_file.readlines())
    input_data_file.close()
    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    n = int(firstLine[0])
    W = int(firstLine[1])
    
    # rework input to be list of tuples (value, weight)
    things1 = [l.split() for l in lines[1:] if l]
    items = np.array([(int(l[0]), int(l[1])) for l in things1])

    return n, W, items

########################################
## Checking ks_30_0
n, W, items = readData('../data/ks_30_0')

## Output from solver-dp-trim-numpy for ks_30_0 
opt = 99798
inds = '0 0 1 0 1 0 1 0 1 0 1 0 1 0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0'
inds = map(int, inds.split(' '))

## check opt and inds
sum([i * j for i, j in zip(inds, items)])

########################################
## Checking ks_50_0
n, W, items = readData('../data/ks_50_0')

## Output from solver-dp-trim-numpy for ks_50_0 
opt = 142156
inds = '0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0'
inds = map(int, inds.split(' '))

## check opt and inds
sum([i * j for i, j in zip(inds, items)])

#######################################
## Checking ks_100_0
inp = solver.readData('../data/ks_100_0')
res = solver.solve_it(inp)
res = res.split(' ')
res[1] = res[1][2]
res = map(int, res)
opt = res[0]
inds = res[1:]

n,W,items = readData('../data/ks_100_0')
## check opt and inds
sum([i * j for i, j in zip(inds, items)])

#######################################
## Checking ks_100_0
inp = solver.readData('../data/ks_100_0')
res = solver.solve_it(inp)
res = res.split(' ')
res[1] = res[1][2]
res = map(int, res)
opt = res[0]
inds = res[1:]

n,W,items = readData('../data/ks_100_0')
## check opt and inds
sum([i * j for i, j in zip(inds, items)])

#######################################
## Checking ks_500_0
inp = solver.readData('../data/ks_500_0')
res = solver.solve_it(inp)
res = res.split(' ')
res[1] = res[1][2]
res = map(int, res)
opt = res[0]
inds = res[1:]

n,W,items = readData('../data/ks_500_0')
## check opt and inds
sum([i * j for i, j in zip(inds, items)])

#######################################
## ****** FAILED **********************
## Checking ks_200_1
inp = solver.readData('../data/ks_200_1')
res = solver.solve_it(inp)
res = res.split(' ')
res[1] = res[1][2]
res = map(int, res)
opt = res[0]
inds = res[1:]

n,W,items = readData('../data/ks_200_1')
## check opt and inds
sum([i * j for i, j in zip(inds, items)])
sum([i * j for i, j in zip(tst, items)])










#######################################
## Checking ks_1000_0
inp = solver.readData('../data/ks_1000_0')
res = solver.solve_it(inp)
res = res.split(' ')
res[1] = res[1][2]
res = map(int, res)
opt = res[0]
inds = res[1:]

n,W,items = readData('../data/ks_1000_0')
## check opt and inds
sum([i * j for i, j in zip(inds, items)])
