r"""
This module should handle the discretization by means of a kmeans or regspace clustering.
"""
import numpy as np
import mcmm

class cluster(object):
    def __init__(self,data,ctrs = 100,dim = 2):
        """    
        Parameters
        ----------
        data: list of ndarrays, i.e. from mcmm.example.generate_test_data()
        ctrs: integer, specifies the number of centers
        dim : integer, specifies the dimension of the points in data

        ----------
        """
        self.trajectories = data
        self.dimension = dim
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
