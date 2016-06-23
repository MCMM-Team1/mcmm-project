import mcmm
import numpy as np
from nose.tools import assert_true

def checkTolerance(m1,m2,tolerance):
    for i in range(len(m1)):
        for j in range(len(m2)):
            if (np.absolute(m1[i][j]-m2[i][j])>=tolerance):
                return False
    return True

def unitvec(i,n):
    x = np.zeros(n)
    x[i]=1
    return x

def determineStep(vec,prob):
    k = 0
    #print(vec)
    #print(prob)
    while(prob > 0.0001):
        #print (vec[k])
        prob -= vec[k]
        #print(prob)
        k+=1
    return k-1

def test_(non)revTmatrix(tolerance = 0.05):
    # test 1: create countmatrix by scaling 
    t1 = np.array([[0.853 , 0.147 , 0 , 0],[0.132 , 0.868 , 0 , 0],[0 , 0 , 0.8 , 0.2],[0 , 0 , 0.2 , 0.8]])
    t2 = np.array([[0.5 , 0.5],[0 , 1]])
    t3 = np.array([[1 , 0 , 0],[0 , 1 , 0],[0 , 0 , 1]])
    t4 = np.array([[0.7,0.3,0,0,0],[0.3,0.7,0,0,0],[0.1,0,0.3,0.5,0.1],[0,0.1,0.5,0.2,0.2],[0,0.1,0.1,0.5,0.3]])
    t5 = np.array([[0.21,0.19,0.212,0.188,0.2],[0.163,0.222,0.178,0.213,0.224],[0.234,0.251,0.117,0.146,0.252],[0.163,0.222,0.178,0.213,0.224],[0.2,0.2,0.2,0.2,0.2]])
    
    c1 = 10000 * t1
    c2 = 10000 * t2
    c3 = 10000 * t3
    c4 = 10000 * t4
    c5 = 10000 * t5
    
    #non reversible
    print(t1)
    print(mcmm.countmatrixTransitionmatrix.nonrevTmatrix(c1))
    assert_true(checkTolerance(t1,mcmm.countmatrixTransitionmatrix.nonrevTmatrix(c1),tolerance))
    assert_true(checkTolerance(t2,mcmm.countmatrixTransitionmatrix.nonrevTmatrix(c2),tolerance))
    assert_true(checkTolerance(t3,mcmm.countmatrixTransitionmatrix.nonrevTmatrix(c3),tolerance))
    assert_true(checkTolerance(t4,mcmm.countmatrixTransitionmatrix.nonrevTmatrix(c4),tolerance))
    assert_true(checkTolerance(t5,mcmm.countmatrixTransitionmatrix.nonrevTmatrix(c5),tolerance))
    
    #reversible
    #print(t1)
    #print(mcmm.countmatrixTransitionmatrix.revTmatrix(c1))
    assert_true(checkTolerance(t1,mcmm.countmatrixTransitionmatrix.revTmatrix(c1),tolerance))
    assert_true(checkTolerance(t2,mcmm.countmatrixTransitionmatrix.revTmatrix(c2),tolerance))
    assert_true(checkTolerance(t3,mcmm.countmatrixTransitionmatrix.revTmatrix(c3),tolerance))
    assert_true(checkTolerance(t4,mcmm.countmatrixTransitionmatrix.revTmatrix(c4),tolerance))
    assert_true(checkTolerance(t5,mcmm.countmatrixTransitionmatrix.revTmatrix(c5),tolerance))
    
    
    # test 2: create trajectoy from a irreducible transition matrix,
    # calculate the count matrix and check the transitionmatrix obtained
    t6 = np.array([[0,1,0],[0,0,1],[1,0,0]])
    t7 = np.array([[0.7,0.1,0.1,0.1],[0.1,0.7,0.1,0.1],[0.15,0.15,0.55,0.15],[0.15,0.15,0.15,0.55]])
    t8 = np.array([[0.3,0.2,0.3,0.2],[0.05,0.8,0.05,0.1],[0.4,0.1,0.3,0.2],[0.25,0.25,0.25,0.25]])
    
    #create trajectories of length 10000, start once in each state
    t61 = np.zeros(10000)
    t62 = np.zeros(10000)
    t63 = np.zeros(10000)
    t61[0]=0
    t62[0]=1
    t63[0]=2
    
    t71 = np.zeros(10000)
    t72 = np.zeros(10000)
    t73 = np.zeros(10000)
    t74 = np.zeros(10000)
    t71[0]=0
    t72[0]=1
    t73[0]=2
    t74[0]=3
    
    t81 = np.zeros(10000)
    t82 = np.zeros(10000)
    t83 = np.zeros(10000)
    t84 = np.zeros(10000)
    t81[0]=0
    t82[0]=1
    t83[0]=2
    t84[0]=3
    
    steps = np.random.random(10000)
    for i in range(9999):
        nextProb =  np.dot(unitvec(t61[i],3),t6)
        t61[i+1] = determineStep(nextProb,steps[i])
    steps = np.random.random(10000)
    for i in range(9999):
        nextProb =  np.dot(unitvec(t62[i],3),t6)
        t62[i+1] = determineStep(nextProb,steps[i])
    steps = np.random.random(10000)
    for i in range(9999):
        nextProb =  np.dot(unitvec(t63[i],3),t6)
        t63[i+1] = determineStep(nextProb,steps[i])
        
    steps = np.random.random(10000)
    for i in range(9999):
        nextProb =  np.dot(unitvec(t71[i],4),t7)
        t71[i+1] = determineStep(nextProb,steps[i])
    steps = np.random.random(10000)
    for i in range(9999):
        nextProb =  np.dot(unitvec(t72[i],4),t7)
        t72[i+1] = determineStep(nextProb,steps[i])
    steps = np.random.random(10000)
    for i in range(9999):
        nextProb =  np.dot(unitvec(t73[i],4),t7)
        t73[i+1] = determineStep(nextProb,steps[i])
    steps = np.random.random(10000)
    for i in range(9999):
        nextProb =  np.dot(unitvec(t74[i],4),t7)
        t74[i+1] = determineStep(nextProb,steps[i])
        
    steps = np.random.random(10000)
    for i in range(9999):
        nextProb =  np.dot(unitvec(t81[i],4),t8)
        t81[i+1] = determineStep(nextProb,steps[i])
    steps = np.random.random(10000)
    for i in range(9999):
        nextProb =  np.dot(unitvec(t82[i],4),t8)
        t82[i+1] = determineStep(nextProb,steps[i])
    steps = np.random.random(10000)
    for i in range(9999):
        nextProb =  np.dot(unitvec(t83[i],4),t8)
        t83[i+1] = determineStep(nextProb,steps[i])
    steps = np.random.random(10000)
    for i in range(9999):
        nextProb =  np.dot(unitvec(t84[i],4),t8)
        t84[i+1] = determineStep(nextProb,steps[i])
        
        
    #create the count matrices
    c61 = mcmm.trajCount.slidingWindowCount(t61,1)
    c62 = mcmm.trajCount.slidingWindowCount(t62,1)
    c63 = mcmm.trajCount.slidingWindowCount(t63,1)
    
    c71 = mcmm.trajCount.slidingWindowCount(t71,1)
    c72 = mcmm.trajCount.slidingWindowCount(t72,1)
    c73 = mcmm.trajCount.slidingWindowCount(t73,1)
    c74 = mcmm.trajCount.slidingWindowCount(t74,1)
    
    c81 = mcmm.trajCount.slidingWindowCount(t81,1)
    c82 = mcmm.trajCount.slidingWindowCount(t82,1)
    c83 = mcmm.trajCount.slidingWindowCount(t83,1)
    c84 = mcmm.trajCount.slidingWindowCount(t84,1)
    
    assert_true(checkTolerance(t6,mcmm.countmatrixTransitionmatrix.nonrevTmatrix(c61),tolerance))
    assert_true(checkTolerance(t6,mcmm.countmatrixTransitionmatrix.nonrevTmatrix(c62),tolerance))
    assert_true(checkTolerance(t6,mcmm.countmatrixTransitionmatrix.nonrevTmatrix(c63),tolerance))
    
    
    
    assert_true(checkTolerance(t7,mcmm.countmatrixTransitionmatrix.nonrevTmatrix(c71),tolerance))
    assert_true(checkTolerance(t7,mcmm.countmatrixTransitionmatrix.nonrevTmatrix(c72),tolerance))
    assert_true(checkTolerance(t7,mcmm.countmatrixTransitionmatrix.nonrevTmatrix(c73),tolerance))
    assert_true(checkTolerance(t7,mcmm.countmatrixTransitionmatrix.nonrevTmatrix(c74),tolerance))
    
    assert_true(checkTolerance(t8,mcmm.countmatrixTransitionmatrix.nonrevTmatrix(c81),tolerance))
    assert_true(checkTolerance(t8,mcmm.countmatrixTransitionmatrix.nonrevTmatrix(c82),tolerance))
    assert_true(checkTolerance(t8,mcmm.countmatrixTransitionmatrix.nonrevTmatrix(c83),tolerance))
    assert_true(checkTolerance(t8,mcmm.countmatrixTransitionmatrix.nonrevTmatrix(c84),tolerance))
    
