# SEND MORE MONEY with ortools
from ortools.constraint_solver import pywrapcp

def main():
    """
      SEND
    + MORE
    ______
     MONEY
    """

    # Create the solver
    solver = pywrapcp.Solver('Send More Money')
    
    # Make the variables
    LD = [solver.IntVar(0, 9, 'LD[%i]' % i) for i in range(0,8)]
    S,E,N,D,M,O,R,Y = LD

    # constraints
    solver.Add(solver.AllDifferent(LD))
    solver.Add(S >= 1)
    solver.Add(M >= 1)
    
    solver.Add(D+10*N+100*E+1000*S+
               E+10*R+100*O+1000*M
               == Y+10*E+100*N+1000*O+10000*M)

    # search and result
    db = solver.Phase(LD,
                      solver.INT_VAR_SIMPLE,
                      solver.INT_VALUE_SIMPLE)

    solver.NewSearch(db)

    num_solutions = 0
    str = "SENDMORY"
    while solver.NextSolution():
        num_solutions += 1
        for (letter, val) in [(str[i], LD[i].Value()) for i in range(len(LD))]:
            print "{}: {}".format(letter, val)
        print

    solver.EndSearch()
    
    print
    print "num_solutions: ", num_solutions
    print "failures: ", solver.Failures()
    print "branches: ", solver.Branches()
    print "WallTime: ", solver.WallTime()

if __name__ == '__main__':
    main()

