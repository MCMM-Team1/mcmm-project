import numpy as np
import scipy as sp
from scipy import spatial

#@np.vectorize
def _computeSquareDistancesToClusters(clusters,point):
    
    temppoint = np.array([point])
    return sp.spatial.distance.cdist(clusters,temppoint,'sqeuclidean')
    
    #clusters has to be a list of points!!!
    temp = np.zeros(len(clusters), dtype=np.float)
    counter = 0
    for x in clusters:
        temp[counter] = np.linalg.norm(np.subtract(x,point))
        counter +=1
    return temp #np.min(np.array([(np.linalg.norm(np.subtract(x,point)) for x in clusters)]))

def _initialization(traj,k):
    distances = np.zeros(len(traj), dtype=np.float)
    r = np.random.randint(0,len(traj))
    # the r'th element is the first cluster center, chosen uniformly at random
    clusters = np.array([traj[r]], dtype=np.float)
    for l in range(1,k):
        for i in range(len(traj)):
            distances[i]=np.min(_computeSquareDistancesToClusters(clusters,traj[i]))
        #choose next cluster point
        nextClusterPoint = _chooseNextClusterPoint(distances)
        clusters = np.concatenate([clusters,np.array([traj[nextClusterPoint]])])
    return clusters

def KMeans(data,dim=2,k=100,tolerance=0.005):
    """
    Parameters
    ----------
    data      : list of numpy ndarrays, list of trajectories, possibly of different lenghts
    dim       : int, the dimension of the trajectories
    k         : int, the number of clusters
    tolerance : float, defines when clusterpoints "dont change", in max-norm
   
    Return
    ------
    distrajs : list of numpy ndarrays, list of discrete trajectories
    centers : numpy ndarray, all the cluster centers
   
    For each trajectory in the given data, do a kmeans(++) algorithm
    to find the clusters and output discrete trajectories.
    Therefore call a kmeans subprocedure.
    """
    superData = np.concatenate(data)
    if len(superData) < k:
        k = len(superData)
        print("reduced number of cluster to the overall number of data")
    allClusters = _initialization(superData,k)
    #result = np.empty((len(data),len(data[0])))
    result = []

    
    #print("initialisation done")
    
    while True:
        #print("entered while loop (again)")
        allClustersOld = allClusters.copy()
        helpme = np.zeros(len(superData),dtype=np.int)
        for c in range(len(superData)):
            helpme[c] = np.argmin(_computeSquareDistancesToClusters(allClusters,superData[c]))
        countSize = np.zeros(k, dtype=np.int)
        countMean = np.zeros((k,dim), dtype=np.float)
        for c in range(len(superData)):
            countSize[helpme[c]] += 1
            countMean[helpme[c]] += superData[c]
        for i in range(k):
            allClusters[i] = np.multiply((1.0/countSize[i]) , countMean[i])
        if np.max(allClusters - allClustersOld) < tolerance and np.min(allClusters - allClustersOld) > (-1)*tolerance:
            break
    """
    helpcounter = 0
    for c1 in range(len(data)):
        for c2 in range(len(data[c1])):
            result[c1][c2] = helpme[helpcounter]
            
            helpcounter += 1 """
    helpcounter = 0
    for c1 in range(len(data)):
        hilfresult = np.empty(len(data[c1]))
        for c2 in range(len(data[c1])):
            hilfresult[c2] = helpme[helpcounter]
            
            helpcounter += 1
        result.append(hilfresult)
    #_result = [result[i, :] for i in range(result.shape[0])]
    #return (_result,allClusters)
    return (result,allClusters)
        
    

def _chooseNextClusterPoint(distances):
    #distances: list of floats
    #temp = np.array([d * d for d in distances])
    total = np.sum(distances)
    rand = np.random.uniform(0,total)
    result = 0
    while(rand >= 0):
        rand -= distances[result]
        if(rand < 0):
            return result
        result += 1
    return -1
