import numpy as np
import scipy.linalg 
import matplotlib.pyplot as plt



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

class msm(object):  
    def __init__(self,P):
        """Initial step: Checks if the matrix is stochastic and quadratic"""
        for i in range(len(P[0])):
            if (abs(np.sum(P[i])-1>1.0E-8)):
                print "Not stochastic"
        if (len(P[0])!=len(P[1])):
            print "Not quadratic"
    
    
    def eigAnalysis(self,P):
        """Calculates eigenvalus and eigevectors both right and left.
        
        Input: Stochastic (int(n),int(n)) matrix.
        
        Output: Array with eigenvalus sorted in descending order.
                Left and right eigenvectors normlaized in the 2-norm v[:,i].
                Stationary distribution as probability distribution as an array."""
        eigval , eigvec_right , eigvec_left = scipy.linalg.eig(P,left=True,right=True)

        eigval = np.real(eigval)
        eigvec_right = np.real(eigvec_right)
        eigvec_left = np.real(eigvec_left)

        idx = np.argsort(abs(eigval))[::-1]

        eigval = eigval[idx]
        eigvec_left = eigvec_left[:,idx]
        eigvec_right = eigvec_right[:,idx]
        
        self.eigval = eigval
        self.eigvec_left = eigvec_left 
        self.eigvec_right = eigvec_right
        self.stationary = eigvec_right[:,0]/np.sum(eigvec_right[:,0])
        return self
    
    def visualizeMatrix(self,P):
        """Plots the transition matrix as a 2D plot"""
        fig = plt.figure(figsize=(6, 3.2))
        ax = fig.add_subplot(111)
        ax.set_title('colorMap')
        plt.imshow(P)
        ax.set_aspect('equal')

        cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
        cax.get_xaxis().set_visible(False)
        cax.get_yaxis().set_visible(False)
        cax.patch.set_alpha(0)
        cax.set_frame_on(False)
        plt.colorbar(orientation='vertical')
        
        return plt.show()
