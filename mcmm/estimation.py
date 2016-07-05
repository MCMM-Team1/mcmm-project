r"""
This module should handle the transition counting and transition matrix estimation.
"""
import numpy as np
import mcmm

class estimate(object):
    def __init__(self,distraj):
        """    
        Parameters
        ----------
        distraj: ndarraylike, should be created by cluster.disctrajectories()  in clustering        
        ----------
        """
        self._transitionMatrix = None
        self._countMatrix = None
        self.disctrajectories = distraj

    @property
    def countMatrix(self):
        if self._countMatrix is None:
             pass #create a count matrix by using self.disctrajectories
        return self._countMatrix

    @property
    def transitionMatrix(self):
        if  self._transitionMatrix is None:
             pass #create a transition matrix by using self.countMatrix
        return  self._transitionMatrix


