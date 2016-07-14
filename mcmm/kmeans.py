import numpy as np
import scipy as sp
from scipy import spatial

#@np.vectorize
def _computeSquareDistancesToClusters(clusters,points):
    #gets a list of clusters and a list of points, uses a vectorized scipy function which returns
    #an array with all the squared distances.
    return sp.spatial.distance.cdist(clusters,points,'sqeuclidean')
    

def _initialization(traj,k):
    #use the k++ initialisation
    distances = np.zeros(len(traj), dtype=np.float)
    r = np.random.randint(0,len(traj))
    # the r'th element is the first cluster center, chosen uniformly at random
    clusters = np.array([traj[r]], dtype=np.float)
    for l in range(1,k):
        distancesmatrix=_computeSquareDistancesToClusters(clusters,traj)
        distances = np.amin(distancesmatrix,axis = 0)  #returns the indices of the nearest cluster points for every data point
        nextClusterPoint = _chooseNextClusterPoint(distances)        #choose next cluster point
        clusters = np.concatenate([clusters,np.array([traj[nextClusterPoint]])])
    return clusters

def KMeans(data,dim=2,k=100,tolerance=0.01):
    """
    Parameters
    ----------
    data      : list of numpy ndarrays, represents the list of trajectories, possibly of different lenghts
    dim       : int, the dimension of the trajectories
    k         : int, the number of clusters
    tolerance : float, defines when clusterpoints "dont change", in max-norm
   
    Return
    ------
    distrajs : list of numpy ndarrays, represents the list of discrete trajectories
    centers : numpy ndarray, all the cluster centers
    """
    
    superData = np.concatenate(data)
    #do kmeans for all the data points at once
    
    if len(superData) < k:
        k = len(superData)
        print("reduced number of cluster to the overall number of data")
        #the number of clusters should be smaller than the number of data points
    
    allClusters = _initialization(superData,k)
    print("initialisation done")

    result = []   #this is the list which will later contain the discrete trajectories
    nums = 1;
    
    while True:
        if nums == 1:
            print("entered while loop one time.")
        else:
            print("{} {} {}".format("entered while loop ",nums," times."))
        nums += 1
        allClustersOld = allClusters.copy()
        helpme = np.zeros(len(superData),dtype=np.int)
        #for c in range(len(superData)):
         #   helpme[c] = np.argmin(_computeSquareDistancesToClusters(allClusters,superData[c]))
        helpme = np.argmin(_computeSquareDistancesToClusters(allClusters,superData),axis = 0)
        #print("helpme",helpme)
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
        hilfresult = np.empty(len(data[c1]),dtype=int)
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
