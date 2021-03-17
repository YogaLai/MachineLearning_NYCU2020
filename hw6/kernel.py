import numpy as np
from scipy.spatial.distance import pdist,squareform

def kernel(X,gamma=[1,1]):
    n=len(X)
    S=np.zeros((n,2))
    for i in range(n):
        S[i]=[i//100,i%100]
    rbf1=squareform((gamma[0]*-pdist(S,'sqeuclidean')))
    rbf2=squareform((gamma[1]*-pdist(X,'sqeuclidean')))
    return rbf1*rbf2