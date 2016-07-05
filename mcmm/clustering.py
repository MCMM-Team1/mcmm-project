r"""
This module should handle the discretization by means of a kmeans or regspace clustering.
"""
import numpy as np
import mcmm

class cluster(object):
    def __init__(self,data,dim = 2,ctrs = 100):
        """    
        Parameters
        ----------
        data: ndarraylike, i.e. from mcmm.example.generate_test_data()
        dim : integer, specifies the dimension of the points in data
        ctrs: integer, specifies the number of centers
        ----------
        """
        self.trajectories = data
        self.dimension = dim
        self.centers = ctrs
        self._disctrajectories = None

    @property
    def disctrajectories(self):
        if self._disctrajectories is None:
             self._disctrajectories = mcmm.kmeans.KMeans(self.trajectories,self.dimension,self.centers)
        return self._disctrajectories

