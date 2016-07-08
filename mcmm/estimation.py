r"""
This module should handle the transition counting and transition matrix estimation.
"""
import numpy as np
import mcmm

class estimate(object):
    def __init__(self,distraj,tau,k):
        """    
        Parameters
        ----------
        distraj: ndarraylike, should be created by cluster.disctrajectories()  in clustering  
        tau : time lag  
        k : number of cluster centers = number of discrete states    
        ----------
        """
        self._transitionMatrix = None
        self._countMatrix = None
        self.disctrajectories = distraj
        self.tau = tau
        self.k = k
        self._dictionaryReducedMatrix = None
        self._reducedCountMatrix = None

    @property
    def reducedCountMatrix(self):
        if self._reducedCountMatrix is None:
            pass #calculate the reduced countmatrix as well as the dictionary that maps the small matrix to the big one
                 #use kosaraju (from the msm class) for this

    @property
    def countMatrix(self):
        if self._countMatrix is None:
             self._countMatrix = mcmm.trajCount.slidingWindowCountXL(self.disctrajectories,self.tau,self.k)
        return self._countMatrix

    @property
    def transitionMatrix(self):                    #transition matrix is always reduced
        if  self._transitionMatrix is None:
             pass #create a transition matrix by using self.countMatrix
        return  self._transitionMatrix


