#!/usr/bin/env python
import os
import os.path
from scipy import *
from pylab import *

def Distance(R1, R2):
    return sqrt((R1[0]-R2[0])**2+(R1[1]-R2[1])**2)

def TotalDistance(city, R):
    dist=0
    for i in range(len(city)-1):
        dist += Distance(R[city[i]],R[city[i+1]])
    dist += Distance(R[city[-1]],R[city[0]])
    return dist
    
def reverse(city, n):
    nct = len(city)
    nn = (1+ ((n[1]-n[0]) % nct))/2 # half the lenght of the segment to be reversed
    # the segment is reversed in the following way n[0]<->n[1], n[0]+1<->n[1]-1, n[0]+2<->n[1]-2,...
    # Start at the ends of the segment and swap pairs of cities, moving towards the center.
    for j in range(nn):
        k = (n[0]+j) % nct
        l = (n[1]-j) % nct
        (city[k],city[l]) = (city[l],city[k])  # swap
    
def transpt(city, n):
    nct = len(city)
    
    newcity=[]
    # Segment in the range n[0]...n[1]
    for j in range( (n[1]-n[0])%nct + 1):
        newcity.append(city[ (j+n[0])%nct ])
    # is followed by segment n[5]...n[2]
    for j in range( (n[2]-n[5])%nct + 1):
        newcity.append(city[ (j+n[5])%nct ])
    # is followed by segment n[3]...n[4]
    for j in range( (n[4]-n[3])%nct + 1):
        newcity.append(city[ (j+n[3])%nct ])
    return newcity

def Plot(city, R, dist):
    # Plot
    Pt = [R[city[i]] for i in range(len(city))]
    Pt += [R[city[0]]]
    Pt = array(Pt)
    title('Total distance='+str(dist))
    plot(Pt[:,0], Pt[:,1], '-o')
    show()


####################################################################################
##                                     I/O
def load_cached_path(filename):
    """Load cached data if available"""
    fp = open(filename, 'r')
    data = ''.join(fp.readlines())
    fp.close()
    lines = data.split()
    return float(lines[0]), map(int, lines[1:])


def output(dist, city):
    output_data = str(dist) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, city))
    return output_data


def save(dist, city):
    """cache path"""
    filename = '/home/noah/class/discopt/tsp/testing/solutions/' +\
               'rev_trans_anneal_' + str(ncity) + '.data'
    fp = open(filename, 'w')
    fp.write("%s " % dist)
    for p in city:
        fp.write("%s " % p)
    fp.close()


import sys

if __name__=='__main__':
    # ncity = 100        # Number of cities to visit
    R = []
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        input_data_file = open(file_location, 'r')
        input_data = ''.join(input_data_file.readlines())
        input_data_file.close()
        lines = input_data.split('\n')
        ncity = int(lines[0])
        print "TSP with {} cities".format(ncity)
        # load points
        for i in range(1, ncity + 1):
            line = lines[i]
            parts = line.split()
            R.append( [float(parts[0]), float(parts[1])] )
        R = array(R)

    else:
        # Choose random coordinates
        for i in range(ncity):
            R.append( [rand(),rand()] )
        R = array(R)

    maxTsteps = 50    # Temperature is lowered not more than maxTsteps
    Tstart = 11        # Starting temperature - has to be high enough
    fCool = 0.96       # Factor to multiply temperature at each cooling step
    maxSteps = 10000*ncity     # Number of steps at constant temperature
    maxAccepted = 10*ncity   # Number of accepted steps at constant temperature

    Preverse = 0.5      # How often to choose reverse/transpose trial move

    # The index table -- the order the cities are visited.
    city = range(ncity)

    # check for cached solution
    best = 2**30
    fname = '/home/noah/class/discopt/tsp/testing/solutions/' +\
                   'rev_trans_anneal_' + str(ncity) + '.data'
    if os.path.isfile(fname):
        if len(sys.argv) > 2 and sys.argv[2] == "reuse":
            best, city = load_cached_path(fname)
        else:
            best, path = load_cached_path(fname)
    print "Initial best tour: {}".format(best)

    # Distance of the travel at the beginning
    dist = TotalDistance(city, R)

    # Stores points of a move
    n = zeros(6, dtype=int)
    nct = len(R) # number of cities
    
    T = Tstart # temperature

    # Plot(city, R, dist)
    
    for t in range(maxTsteps):  # Over temperature

        accepted = 0
        for i in range(maxSteps): # At each temperature, many Monte Carlo steps
            
            while True: # Will find two random cities sufficiently close by
                # Two cities n[0] and n[1] are choosen at random
                n[0] = int((nct)*rand())     # select one city
                n[1] = int((nct-1)*rand())   # select another city, but not the same
                if (n[1] >= n[0]): n[1] += 1   #
                if (n[1] < n[0]): (n[0],n[1]) = (n[1],n[0]) # swap, because it must be: n[0]<n[1]
                nn = (n[0]+nct -n[1]-1) % nct  # number of cities not on the segment n[0]..n[1]
                if nn>=3: break
        
            # We want to have one index before and one after the two cities
            # The order hence is [n2,n0,n1,n3]
            n[2] = (n[0]-1) % nct  # index before n0  -- see figure in the lecture notes
            n[3] = (n[1]+1) % nct  # index after n2   -- see figure in the lecture notes
            
            if Preverse > rand(): 
                # Here we reverse a segment
                # What would be the cost to reverse the path between city[n[0]]-city[n[1]]?
                de = Distance(R[city[n[2]]],R[city[n[1]]]) + Distance(R[city[n[3]]],R[city[n[0]]]) - Distance(R[city[n[2]]],R[city[n[0]]]) - Distance(R[city[n[3]]],R[city[n[1]]])
                
                if de<0 or exp(-de/T)>rand(): # Metropolis
                    accepted += 1
                    dist += de
                    reverse(city, n)
            else:
                # Here we transpose a segment
                nc = (n[1]+1+ int(rand()*(nn-1)))%nct  # Another point outside n[0],n[1] segment. See picture in lecture nodes!
                n[4] = nc
                n[5] = (nc+1) % nct
        
                # Cost to transpose a segment
                de = -Distance(R[city[n[1]]],R[city[n[3]]]) - Distance(R[city[n[0]]],R[city[n[2]]]) - Distance(R[city[n[4]]],R[city[n[5]]])
                de += Distance(R[city[n[0]]],R[city[n[4]]]) + Distance(R[city[n[1]]],R[city[n[5]]]) + Distance(R[city[n[2]]],R[city[n[3]]])
                
                if de<0 or exp(-de/T)>rand(): # Metropolis
                    accepted += 1
                    dist += de
                    city = transpt(city, n)
                    
            if accepted > maxAccepted: break

        # Plot
        # Plot(city, R, dist)
            
        print "T=%10.5f , distance= %10.5f , accepted steps= %d" %(T, dist, accepted)
        T *= fCool             # The system is cooled down
        # if accepted == 0: break  # If the path does not want to change any more, we can stop
        if dist < best:
            os.system('paplay $BEEP')
            print "saving..."
            save(dist, city)
            best = dist

    if dist < best:
        save(dist, city)
    print output(dist, city)
    Plot(city, R, dist)
    os.system('paplay $BEEP')
