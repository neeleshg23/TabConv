import itertools
import numpy as np
from sklearn import cluster
from scipy import signal
from sklearn.cluster import KMeans
from joblib import Memory
_memory = Memory('.', verbose=0)

@_memory.cache
#def kmeans(X, k, max_iter=16, init='kmc2', return_sse=False):
def kmeans(X, k, max_iter=16, init='subspaces', return_sse=False):
    X = X.astype(np.float32)

    # handle fewer nonzero rows than centroids (mostly just don't choke
    # if X all zeros, which happens when run in PQ with tiny subspaces)
    rowsums = X.sum(axis=1)
    nonzero_mask = rowsums != 0
    nnz_rows = np.sum(nonzero_mask)
    if nnz_rows < k:
        print("X.shape: ", X.shape)
        print("k: ", k)
        print("nnz_rows: ", nnz_rows)

        centroids = np.zeros((k, X.shape[1]), dtype=X.dtype)
        labels = np.full(X.shape[0], nnz_rows, dtype=np.int)
        if nnz_rows > 0:  # special case, because can't have slice of size 0
            # make a centroid out of each nonzero row, and assign only those
            # rows to that centroid; all other rows get assigned to next
            # centroid after those, which is all zeros
            centroids[nnz_rows] = X[nonzero_mask]
            labels[nonzero_mask] = np.arange(nnz_rows)
        if return_sse:
            return centroids, labels, 0
        return centroids, labels

    # if k is huge, initialize centers with cartesian product of centroids
    # in two subspaces
    sqrt_k = int(np.ceil(np.sqrt(k)))
    if k >= 16 and init == 'subspaces':
        print("kmeans: clustering in subspaces first; k, sqrt(k) ="
              " {}, {}".format(k, sqrt_k))
        _, D = X.shape
        #centroids0, _ = kmeans(X[:, :D//2], sqrt_k, max_iter=1)
        #centroids1, _ = kmeans(X[:, D//2:], sqrt_k, max_iter=1)


        kmeans1 = KMeans(n_clusters=sqrt_k, random_state=0).fit(X[:, :D//2])
        kmeans2 = KMeans(n_clusters=sqrt_k, random_state=0).fit(X[:, D//2:])
        centroids0, centroids1 = kmeans1.cluster_centers_, kmeans2.cluster_centers_

        seeds = np.empty((sqrt_k * sqrt_k, D), dtype=np.float32)
        for i in range(sqrt_k):
            for j in range(sqrt_k):
                row = i * sqrt_k + j
                seeds[row, :D//2] = centroids0[i]
                seeds[row, D//2:] = centroids1[j]
        seeds = seeds[:k]  # rounded up sqrt(k), so probably has extra rows

        '''
        # the naive k-means work but very slow, the above is a somework faster way
        # kmc2 is fast, but cannot be installed in my env
        k_means_res = KMeans(n_clusters=k, random_state=0).fit(X)
        seeds = k_means_res.cluster_centers_.astype(np.float32)
        '''

    else:
        raise ValueError("init parameter must be one of {'kmc2', 'subspaces'}")

    # this is kindly a fine-tune step
    est = cluster.MiniBatchKMeans(
        k, init=seeds, max_iter=max_iter, n_init=1).fit(X)
    if return_sse:
        return est.cluster_centers_, est.labels_, est.inertia_
    return est.cluster_centers_, est.labels_