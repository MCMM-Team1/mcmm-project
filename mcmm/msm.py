import numpy as np
import scipy.linalg as alg
import matplotlib.pyplot as plt
from msmtools.analysis import pcca as _pcca
import mcmm

def rawScatter(raw_data):

    """Shows scatterplot of data
    
    Parameters
    ----------
    raw data    : trajectories from mcmm.example.generate_test_data
    ---------- 
    
    Return
    ----------
    scatter plot: Returns one plot for 2-dimensional data. 3 plots for three dimensional data.
    ----------
    """

    #Defines the number of plottet points in the scatter plot, nPointsPlot (2000 is good)

    nTraj, trajLength, nDim= np.shape(raw_data)

    nPointsPlot = 2000
    if (trajLength<nPointsPlot):
        a=1
    else: 
        a = trajLength/nPointsPlot

    if (nDim==3):
        '''Scatter plot of raw data.  For 3 dimensional data, the scatter plots are for x,y x,z and y,z.'''
        fig, ax = plt.subplots(1, nDim, figsize=(nDim * 5, 5))
        for rd in raw_data:
            ax[0].scatter(rd[::a, 0], rd[::a, 1], c='grey', s=20)
            format_square(ax[0])
            fig.tight_layout()
        for rd in raw_data:
            ax[1].scatter(rd[::a, 0], rd[::a, 1], c='grey', s=20)
            format_square(ax[1])
            ax[1].set_xlabel(r"$x$ / a.u.")
            ax[1].set_ylabel(r"$z$ / a.u.")
            fig.tight_layout()
        for rd in raw_data:
            ax[2].scatter(rd[::a, 0], rd[::a, 1], c='grey', s=20)
            format_square(ax[2])
            ax[2].set_xlabel(r"$y$ / a.u.")
            ax[2].set_ylabel(r"$z$ / a.u.")
            return fig.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(5, 5))
        for rd in raw_data:
            ax.scatter(rd[::a, 0], rd[::a, 1], c='grey', s=20)
            format_square(ax)
            return fig.tight_layout()



def impliedTimescales(trajs,lagtimes,plotboolean=True):
    """Calculates the implied timescales for different lagtimes and plots it if not set to False
    
    Parameters
    ----------
    trajs       : matrixlike ; discretized trajectories (more than one)
    lagtimes    : arraylike ; list of different lagtimes
    plotboolean : boolean ; default is True for plotting
    ---------- 
    
    Return
    ----------
    timescales: matrixlike ; each column referring to one lagtime and
                each row referring to one eigenvalue of the transitionmatrix
    ----------
    """
    
    if np.max(trajs)>10:
        n=10
    else:
        n=np.max(trajs)
        
    eigval=np.zeros([n,len(lagtimes)])
    timescales=np.zeros([n,len(lagtimes)])
    for i in range(len(lagtimes)):
        lag=mcmm.trajCount.slidingWindowCountXL(trajs,lagtimes[i])
        lag=mcmm.countmatrixTransitionmatrix.revTmatrix(lag)
        lag=mcmm.msm.MSM(lag)
        eigval[:,i]=lag.eigvalues[1:n+1]
        for j in range(n):
            timescales[j,i]=-1./np.log(abs(eigval[j,i]))
    
    if plotboolean:
        fig=plt.figure(figsize=[5,5])
        for i in range(len(timescales[:,0])):
            plt.loglog(lagtimes,timescales[i,:],'-o')
        plt.xlabel('lag time / steps')
        plt.ylabel('timescale / steps')
        fig.tight_layout()
        plt.show()        
    
    return timescales


def DFS(D,v,E,label = []):
	"""
	Depth-First-Search: Finds all vertices which are accessible from node v
	
	Parameters
	----------
	D    : matrixlike, i.e list of lists
	v    : integer, the "starting node" 
	E    : list of discovered nodes
	label: list of nodes which are labeled "True"
	----------
	
	Return
	------
	updated list E
	------
	"""
	
	#label v  as discovered
	if v in label:
		pass
	else:
		label.append(v)
		
	#for all outgoing arcs [v,w] from v do DFS(D,w,E,label) if w is not in label 
	for node in range(len(D[v])):
		if node != v and D[v][node] > 0:
			if node in label:
				pass
			else:
				DFS(D,node,E,label)
	
	#add v to E if it is not already in E
	if v in E:
		pass
	else:
		E.append(v)
	return E


def kosaraju(T):
	"""
	Generates all communication classes of a transition matrix
	
	Parameters
	----------
	T: matrixlike, i.e list of lists
	----------
	
	Return
	------
	List of communication classes
	------
	"""
	V = []
	CC = []
	D = list(T)
	while len(V) < len(D):
		v = [x for x in range(len(D)) if x not in V][0]
		DFS(D,v,V,[])
	Dt = np.transpose(D) # transposed matrix
	while V != []:
		v = V[-1]
		C = DFS(Dt,v,[],label = []) #current communication class
		CC.append(C) #append the current communication class to the output
		for i in range(len(Dt)):                                                                         
			for j in C:                                                                               
				Dt[i][j] = 0 #set columns of current communication class to 0
				Dt[j][i] = 0 #set rows of current communication class to 0
		for i in C:
			V.remove(i)
	return CC

def findLargestCommClass(T):
	"""
	Calls kosaraju and finds the largest communication class of the transition matrix T
	
	Parameters
	----------
	T: matrixlike, i.e list of lists
	----------
	
	Return
	------
	Largeset communication classes of T
	------
	"""
	
	classes = kosaraju(T)
	return max(classes, key=lambda coll: len(coll)) #returns the first maximum communication class
	

class MSM(object):  
	
	
    def __init__(self,P):
        if(not checkStochasticMatrix(P)):
            raise Exception("error: the given matrix t is not stochastic!")
        else:
            self.transition_matrix = P
            self._eigvalues = None
            self._lefteigvectors = None
            self._righteigvectors = None
            self._stationary = None  
    @property
    def stationary(self):
        if self._stationary is None:
            eigvalue,eigvector = alg.eig(self.transition_matrix,left = True,right = False)
            index = np.argmax(eigvalue)
            sol = eigvector[:,index]
            summe = sum(sol)
            self._stationary = sol/summe
        return self._stationary
    
    @property
    def eigvalues(self):
        if self._eigvalues is None:
            eigvalue,eigvectors =  alg.eig(self.transition_matrix,left = True,right = False)
            idx = np.argsort(abs(eigvalue))[::-1]
            eigvalue = eigvalue[idx]
            self._eigvalues = eigvalue
        return self._eigvalues
    
    
    @property
    def lefteigvectors(self):
        if self._lefteigvectors is None:
            eigvalue,eigvectors =  alg.eig(self.transition_matrix,left = True,right = False)
            idx = np.argsort(abs(eigvalue))[::-1]
            eigvectors = eigvectors[:,idx]
            self._lefteigvectors = eigvectors
        return self._lefteigvectors
    
    @property
    def righteigvectors(self):
        if self._righteigvectors is None:
            eigvalue,eigvectors =  alg.eig(self.transition_matrix,left = False,right = True)
            idx = np.argsort(abs(eigvalue))[::-1]
            eigvectors = eigvectors[:,idx]
            self._righteigvectors = eigvectors
        return self._righteigvectors
    
    def pcca(self, numstates):
    	return _pcca(self.transition_matrix , numstates)
    
    
    
    def visualizeMatrix(self):
        """Plots the transition matrix as a 2D plot"""
        fig = plt.figure(figsize=(6, 3.2))
        ax = fig.add_subplot(111)
        ax.set_title('colorMap')
        plt.imshow(self.transition_matrix)
        ax.set_aspect('equal')

        cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
        cax.get_xaxis().set_visible(False)
        cax.get_yaxis().set_visible(False)
        cax.patch.set_alpha(0)
        cax.set_frame_on(False)
        plt.colorbar(orientation='vertical')
        
        return plt.show()
        
    def visualizeEigvalues(self):
        """Plots the 10 biggest Eigenvalues"""
        if len(self.eigvalues)>10:
            n=10
        else:
            n=len(self.eigvalues)
        sorteig=np.sort(self.eigvalues)[::-1]   
        plt.bar(np.arange(n)+0.75,sorteig[:n],0.5)
        plt.xticks(np.arange(n)+1)
        plt.xlabel(r'Index i')
        plt.ylabel(r'Eigenvalue $\lambda_i$')
        plt.title(r'Eigenvalues')
        plt.ylim(0,1.05)
        return plt.show()
        
    def visualizeStationary(self):
        plt.bar(np.arange(len(self.stationary))+0.75,self.stationary,0.5)
        plt.xticks(np.arange(len(self.stationary))+1)
        plt.title(r'Stationary Distribution')
        plt.xlabel(r'index of states')
        plt.ylabel(r'probability')
        return plt.show()
        
        
def checkStochasticMatrix(t):
        
    """
    checks if the given matrix t is row stochastic
     
    Paramters
    ---------
    t : matrix
    
    Return
    ------
    result : true if t is a row stochastic matrix, false otherwise
    """
    numrows = len(t)
    for i in range(numrows):
        if (t.sum(axis=1)[i] > 1.0000001 or t.sum(axis=1)[i] < 0.999999):
            print("if fall")
            print(t.sum(axis=1)[i])
            return False
    return True
  







