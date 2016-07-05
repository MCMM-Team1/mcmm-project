import numpy as np

#@np.vectorize
def _computeDistancesToClusters(clusters,point):
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
            distances[i]=np.min(_computeDistancesToClusters(clusters,traj[i]))
        #choose next cluster point
        nextClusterPoint = _chooseNextClusterPoint(distances)
        clusters = np.concatenate([clusters,np.array([traj[nextClusterPoint]])])
    return clusters

def KMeans(data,dim=2,k=100):
    """
    Parameters
    ----------
    data      : numpy ndarraylike, list of trajectories, all of the same length!!!
    dim       : int, the dimension of the trajectories
    k         : int, the number of clusters
   
    Return
    ------
    distrajs : numpy ndarraylike, list of discrete trajectories
   
    For each trajectory in the given data, do a kmeans(++) algorithm
    to find the clusters and output discrete trajectories.
    Therefore call a kmeans subprocedure.
    """
    superData = np.concatenate(data)
    
    allClusters = _initialization(superData,k)
    result = np.empty((len(data),len(data[0])))
    while True:
        allClustersOld = allClusters
        helpme = np.zeros(len(superData))
        for c in range(len(superData)):
            helpme[c] = np.argmin(_computeDistancesToClusters(allClusters,superData[c]))
        countSize = np.zeros(k, dtype=np.int)
        countMean = np.zeros((k,dim), dtype=np.float)
        for c in range(len(superData)):
            countSize[helpme[c]] += 1
            countMean[helpme[c]] += superData[c]
        for i in range(k):
            allClusters[i] = np.multiply((1.0/countSize[i]) , countMean[i])
        if np.max(allClusters - allClustersOld) < 0.1 and np.min(allClusters - allClustersOld) > -0.1:
            break
    
    helpcounter = 0
    for c1 in range(len(data)):
        for c2 in range(len(data[c1])):
            result[c1][c2] = helpme[helpcounter]
            helpcounter += 1
    return result
        
    

def _chooseNextClusterPoint(distances):
    #distances: list of floats
    temp = np.array([d * d for d in distances])
    total = np.sum(temp)
    rand = np.random.uniform(0,total)
    result = 0
    while(rand >= 0):
        rand -= temp[result]
        if(rand < 0):
            return result
        result += 1
    return -1
