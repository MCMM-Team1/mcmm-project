def pca(data,reducedDim,numberpoints=0):
    """PCA dimension reduction
    Parameters
    ----------    
    data : matrixlike, generate_test_data format
    reducedDim: number of reduced dimensions
    numberpoints: if given reduces the number of points (not really necessary)
    
    Returns
    ---------
    creates x or x,y or x,y,z or ... 2d-array with the rows corresponding to the reduced dimensios
    """
    trajleng=np.shape(data)[1]
    sdata=np.concatenate(data)
    if numberpoints==0:
        new=sdata
    else:
        leng=int(len(sdata)/numberpoints)
        new=sdata[::leng]
    new=np.transpose(new)
    for i in range(len(new)):
        new[:,i]=new[:,i]-np.mean(new[:,i])
    #optionally
    for j in range(len(new)):
        new[:,i]=new[:,i]/np.std(new[:,i])
    u,v,w=np.linalg.svd(np.cov(new))
    cred=np.dot(np.transpose(u[:,np.arange(reducedDim)]),new)
    
    if reducedDim==1:
        u=cred[0]
        u=u[:trajleng]
        return u
    else:   
        u=cred[np.arange(reducedDim)]
        u=u[:,:trajleng]
        return u
