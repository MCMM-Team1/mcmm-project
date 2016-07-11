import numpy as np

def slidingWindowCountXL(trajs,tau,k):
    """
    returns the countmatrix, gets a list of trajectories and tau
    assumes that the discretization labels states, beginning with 0
    
    Paramaters
    ----------
    trajs: list of numpy.ndarray
    tau : integer
    k : integer, number of states (= number of clusters)
    ----------
    
    Return
    ------
    Countmatrix: numpy.ndarray
    ------
    """
    Countmtrx = np.zeros((k,k))
    for traj in trajs:
        Countmtrx += slidingWindowCount(traj,tau,k)
    return Countmtrx

def slidingWindowCount(traj,tau,k):
    """
    returns the countmatrix, gets a trajectory and tau
    assumes that the discretization labels states, beginning with 0
    
    Paramaters
    ----------
    traj: arraylike
    tau : integer
    k : integer, number of states (= number of clusters)
    ----------
    
    Return
    ------
    Countmatrix: numpy.ndarray
    ------
    """
    Countmatrix = np.zeros((k,k))
    T = len(traj) 

    for l in range(int(T - tau)):
        Countmatrix[int(traj[int(l)])][int(traj[int(l+tau)])]+=1
    
    """for i in range(k):
        for j in range(k):
            summe = 0
            for l in range(T- tau ):
                if traj[l] == i and traj[l+tau] == j:
                    summe +=1
            Countmatrix[i][j] = summe"""
    return Countmatrix
