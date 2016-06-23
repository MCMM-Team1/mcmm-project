import numpy as np

def slidingWindowCountXL(trajs,tau):
    """
    returns the countmatrix, gets a list of trajectories and tau
    
    Paramaters
    ----------
    trajs: array of arraylike
    tau : integer
    ----------
    
    Return
    ------
    Countmatrix: matrixlike
    ------
    """
    Countmtrx = np.zeros((n,n))
    for traj in trajs:
        Countmtrx += slidingWindowCount(traj,tau)
    return Countmtrx

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
    n = int(np.max(traj)+ 1)
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
