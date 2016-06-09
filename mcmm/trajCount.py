def slidingWindowCount(traj,tau):
    """
    returns the countmatrix, gets a trajectory and tau
    
    Paramaters
    ----------
    traj: arraylike
    tau : integer
    ----------
    
    Return
    ------
    Countmatrix: matrixlike
    ------
    """
    n = np.max(traj)+ 1
    Countmatrix = np.zeros((n,n))
    T = len(traj) 
    
    for i in range(n):
        for j in range(n):
            summe = 0
            for k in range(T- tau ):
                if traj[k] == i and traj[k+tau] == j:
                    summe +=1
            Countmatrix[i][j] = summe
    return Countmatrix
