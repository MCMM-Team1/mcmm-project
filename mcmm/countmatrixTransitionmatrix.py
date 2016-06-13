# here come the nonrevTmatrix(matrix count) and revTmatrix(matrix count) functions
def revTmatrix(t):
    guess=t / np.sum(t,1)[:,None]
    stat=MCMM(guess)
    stat.eigAnalysis(guess)
    pi=stat.stationary
    x=np.zeros([len(t),len(t)])
    xold=x
    delta=1
    for i in range(len(t)):
        x[i,:]=guess[i,:]*pi[i]
    while (delta < 1e-3):
        xold=x
        for i in range(len(t)):
            for j in range(len(t)):
                x[i,j]=(t[i,j]+t[j,i])/(np.sum(t,1)/np.sum(xold,1)+np.sum(t,0),np.sum(xold,0))
        delta=np.sum(abs(x-xold))
    print (x)
    t=x/ np.sum(x,1)[:,None]
    return t

def nonrevTmatrix(t):
    t=t / np.sum(t,1)[:,None]
    return t
