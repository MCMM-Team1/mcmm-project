r"""
The mcmm package should allow to build Markov state models from trajectory data. Three stages are
necessary: discretization (clustering), estimation, and analysis of the resulting model.
"""

from . import example
from . import msm
from . import countmatrixTransitionmatrix
from . import trajCount
from . import clustering
from . import estimation
from . import kmeans
from . import analysis
from . import pca
