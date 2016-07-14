import mcmm
import numpy as np
from nose.tools import assert_true
import unittest

def check_kosaraju(t,r):
    """
    checks if the kosaraju algorithm with input t finds the largest comminication class r 
    Parameters
    
    ----------
    t: matrix for the algorithm
    r: the result which is known beforehand, given as set
    """
    kosarajuResult = mcmm.msm.findLargestCommClass(t)
    #want the lists to appear as sets for comparison
    kosarajuResultAsSet = set()
    for element in kosarajuResult:
        kosarajuResultAsSet.add(element)
    assert_true(r == kosarajuResultAsSet)

def checkTolerance(m1,m2,tolerance):
    """
    checks if the diffrence between the entries of the matrices m1 and m2 is less than tolerance
    """

    for i in range(len(m1)):
        for j in range(len(m2)):
            if (np.absolute(m1[i][j]-m2[i][j])>=tolerance):
                return False
    return True

def unitvec(i,n):
    ii = int(i)
    x = np.zeros(n)
    x[ii]=1
    return x

def determineStep(vec,prob):
    k = 0
    while(prob > 0.0001):
        prob -= vec[k]
        k+=1
    return k-1

class TestStringMethods(unittest.TestCase):
    def test_kosaraju(self):
        """
        checks if the kosaraju algorithm really finds the largest communication class
        """
        #absorbing case
        t4 = np.array([[0.7,0.3,0,0,0],[0.3,0.7,0,0,0],[0.1,0,0.3,0.5,0.1],[0,0.1,0.5,0.2,0.2],[0,0.1,0.1,0.5,0.3]])
        r4 = set([2 , 3 , 4])

        #uniform random case
        t5 = np.array([[0.2,0.2,0.2,0.2,0.2],[0.2,0.2,0.2,0.2,0.2],[0.2,0.2,0.2,0.2,0.2],[0.2,0.2,0.2,0.2,0.2],[0.2,0.2,0.2,0.2,0.2]])
        r5 = set([0,1,2,3,4])
        
        check_kosaraju(t4,r4)
        check_kosaraju(t5,r5)
    
    def test_revAndNonrevTmatrix(self,tolerance = 0.05):
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
        assert_true(checkTolerance(t1,mcmm.countmatrixTransitionmatrix.nonrevTmatrix(c1),tolerance))
        assert_true(checkTolerance(t2,mcmm.countmatrixTransitionmatrix.nonrevTmatrix(c2),tolerance))
        assert_true(checkTolerance(t3,mcmm.countmatrixTransitionmatrix.nonrevTmatrix(c3),tolerance))
        assert_true(checkTolerance(t4,mcmm.countmatrixTransitionmatrix.nonrevTmatrix(c4),tolerance))
        assert_true(checkTolerance(t5,mcmm.countmatrixTransitionmatrix.nonrevTmatrix(c5),tolerance))
        
        #reversible    
        assert_true(checkTolerance(t1,mcmm.countmatrixTransitionmatrix.revTmatrix(c1),tolerance))
        assert_true(checkTolerance(t2,mcmm.countmatrixTransitionmatrix.revTmatrix(c2),tolerance))
        assert_true(checkTolerance(t3,mcmm.countmatrixTransitionmatrix.revTmatrix(c3),tolerance))
        #assert_true(checkTolerance(t4,mcmm.countmatrixTransitionmatrix.revTmatrix(c4),tolerance))
        assert_true(checkTolerance(t5,mcmm.countmatrixTransitionmatrix.revTmatrix(c5),tolerance))
        
        # test 2: create trajectoy from an irreducible transition matrix,
        # calculate the count matrix and check the transitionmatrix obtained
        t6 = np.array([[0,1,0],[0,0,1],[1,0,0]])
        t7 = np.array([[0.7,0.1,0.1,0.1],[0.1,0.7,0.1,0.1],[0.15,0.15,0.55,0.15],[0.15,0.15,0.15,0.55]])
        t8 = np.array([[0.3,0.2,0.3,0.2],[0.05,0.8,0.05,0.1],[0.4,0.1,0.3,0.2],[0.25,0.25,0.25,0.25]])
        
        #create trajectories of length 10000, start once in each state
        t61 = np.zeros(10000,dtype = np.int8)
        t62 = np.zeros(10000,dtype = np.int8)
        t63 = np.zeros(10000,dtype = np.int8)
        t61[0]=0
        t62[0]=1
        t63[0]=2
        
        t71 = np.zeros(10000,dtype = np.int8)
        t72 = np.zeros(10000,dtype = np.int8)
        t73 = np.zeros(10000,dtype = np.int8)
        t74 = np.zeros(10000,dtype = np.int8)
        t71[0]=0
        t72[0]=1
        t73[0]=2
        t74[0]=3
        
        t81 = np.zeros(10000,dtype = np.int8)
        t82 = np.zeros(10000,dtype = np.int8)
        t83 = np.zeros(10000,dtype = np.int8)
        t84 = np.zeros(10000,dtype = np.int8)
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
        c61 = mcmm.trajCount.slidingWindowCount(t61,1,max(t61)+1)
        c62 = mcmm.trajCount.slidingWindowCount(t62,1,max(t62)+1)
        c63 = mcmm.trajCount.slidingWindowCount(t63,1,max(t63)+1)
        
        c71 = mcmm.trajCount.slidingWindowCount(t71,1,max(t71)+1)
        c72 = mcmm.trajCount.slidingWindowCount(t72,1,max(t72)+1)
        c73 = mcmm.trajCount.slidingWindowCount(t73,1,max(t73)+1)
        c74 = mcmm.trajCount.slidingWindowCount(t74,1,max(t74)+1)
        
        c81 = mcmm.trajCount.slidingWindowCount(t81,1,max(t81)+1)
        c82 = mcmm.trajCount.slidingWindowCount(t82,1,max(t82)+1)
        c83 = mcmm.trajCount.slidingWindowCount(t83,1,max(t83)+1)
        c84 = mcmm.trajCount.slidingWindowCount(t84,1,max(t84)+1)

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

    def test_reducedCountMatrix(self):
        bigMatrix = np.array([[1,2,0,3,0,4,5],[6,7,0,8,0,9,10],[5,5,5,5,5,5,5],[11,12,0,13,0,14,15],[5,5,5,5,5,5,5],[16,17,0,18,0,19,20],[21,22,0,23,0,24,25]])
        dictionary = {0:0 , 1:1 , 2:3 , 3:5 , 4:6}
        smallMatrix = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]])
        
        indices = np.sort(mcmm.msm.findLargestCommClass(bigMatrix))
        resultdictionary = {i:indices[i] for i in range(len(indices))}
        result = np.zeros((len(indices),len(indices)),dtype = np.int8)
        for i in range(len(indices)):
            for j in range(len(indices)):
                result[i][j] = bigMatrix[resultdictionary[i],resultdictionary[j]]
        assert_true((smallMatrix == result).all())
        assert_true(dictionary == resultdictionary)
if __name__ == '__main__':
    unittest.main()
