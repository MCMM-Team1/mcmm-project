# here come the nonrevTmatrix(matrix count) and revTmatrix(matrix count) functions
import numpy as np
import mcmm
def revTmatrix(t):
    guess=t / np.sum(t,1)[:,None]
    stat=mcmm.msm.MSM(guess)
    pi=stat.stationary
    x=np.zeros([len(t),len(t)])
    xold=x
    delta=1
    for i in range(len(t)):
        x[i,:]=guess[i,:]*pi[i]
    while (delta < 1e-8):
        xold=np.copy(x)
        for i in range(len(t)):
            for j in range(len(t)):
                x[i,j]=(t[i,j]+t[j,i])/(np.sum(t[i,:])/np.sum(xold[i,:])+np.sum(t[j,:])/np.sum(xold[j,:]))
        delta=np.sum(abs(x-xold))
    t = np.divide(x,(np.sum(x,1)[:,None])*1.0)    
    return t

def nonrevTmatrix(t):
    for i in range(len(t)):
        var = 0
        for j in range(len(t)):
            var = var + t[i][j]
        for j in range(len(t)):
            if var !=0:
                t[i][j] = t[i][j] / var
    return t
