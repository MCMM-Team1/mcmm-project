r"""
This module should handle the discretization by means of a kmeans or regspace clustering.
"""
import numpy as np
import mcmm

class cluster(object):
    def __init__(self,data,ctrs = 100):
        """    
        Parameters
        ----------
        data: list of ndarrays, i.e. from mcmm.example.generate_test_data
        ctrs: integer, specifies the number of centers

        ----------
        """
        self.trajectories = data
        self.dimension = len(data[0][0])
        self.numberCenters = ctrs
        self._disctrajectories = None
        self._centers = None

    @property
    def disctrajectories(self):
        if self._disctrajectories is None:
             (self._disctrajectories,self._centers) = mcmm.kmeans.KMeans(self.trajectories,self.dimension,self.numberCenters)
        return self._disctrajectories


    @property
    def centers(self):
        if self._disctrajectories is None:
             (self._disctrajectories,self._centers) = mcmm.kmeans.KMeans(self.trajectories,self.dimension,self.numberCenters)
        return self._centers
