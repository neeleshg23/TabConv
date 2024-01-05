import time
# import numpy as cp
import cupy as cp

def _dists_elemwise_sq(x, q):
    diffs = x - q
    return diffs * diffs

def _encode_X_pq(X, codebooks, elemwise_dist_func=_dists_elemwise_sq):
    X = cp.asarray(X)
    codebooks = cp.asarray(codebooks) 
   
    ncentroids, nsubvects, subvect_len = codebooks.shape

    assert X.shape[1] == (nsubvects * subvect_len)

    idxs = cp.empty((X.shape[0], nsubvects), dtype=cp.int32)
    X = X.reshape((X.shape[0], nsubvects, subvect_len))  # from (10000,512) to (10000,2,256)
    for i, row in enumerate(X):
        row = row.reshape((1, nsubvects, subvect_len))
        dists = elemwise_dist_func(codebooks, row)
        dists = cp.sum(dists, axis=2)  # sum at the axis of dimension 256, get (16,2)
        idxs[i, :] = cp.argmin(dists, axis=0)  # select min at 0 axis: the 16 clusters

    return idxs