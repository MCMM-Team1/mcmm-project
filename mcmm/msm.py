import numpy as np

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
