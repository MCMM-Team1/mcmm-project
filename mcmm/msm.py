import numpy as np
import scipy.linalg as alg
import matplotlib.pyplot as plt
from msmtools.analysis import pcca as _pcca


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
            eigvalue,eigvector =  alg.eig(self.transition_matrix,left = True,right = False)
            self._eigvalues = eigvalue
        return self._eigvalues
    
    
    @property
    def lefteigvectors(self):
        if self._lefteigvectors is None:
            eigvalue,eigvector =  alg.eig(self.transition_matrix,left = True,right = False)
            self._lefteigvectors = eigvectors
        return self._lefteigvectors
    
    @property
    def righteigvectors(self):
        if self._righteigvectors is None:
            eigvalue,eigvector =  alg.eig(self.transition_matrix,left = False,right = True)
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
  







