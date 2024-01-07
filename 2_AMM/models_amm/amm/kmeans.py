import torch
import cupy as cp

GPU = None

def set_gpu(gpu):
    global cuKMeans
    global GPU
    GPU = gpu
    cp.cuda.Device(gpu).use()
    from cuml.cluster import KMeans as cuKMeans
    
def get_device():
    return torch.device(f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu')  
    
def kmeans(X, k, max_iter=16, init='subspaces', return_sse=False):
    X = cp.asarray(X, dtype=cp.float32)

    # Handle fewer nonzero rows than centroids
    rowsums = X.sum(axis=1)
    nonzero_mask = rowsums != 0
    nnz_rows = cp.sum(nonzero_mask)
    if nnz_rows < k:
        centroids = cp.zeros((k, X.shape[1]), dtype=X.dtype)
        labels = cp.full(X.shape[0], nnz_rows, dtype=cp.int32)
        if nnz_rows > 0:
            centroids[:nnz_rows] = X[nonzero_mask]
            labels[nonzero_mask] = cp.arange(nnz_rows)
        if return_sse:
            return centroids, labels, 0
        return centroids, labels
    
    seeds = 'k-means++' 
    # Initialization of centroids
    if k >= 16 and init == 'subspaces':
        _, D = X.shape
        sqrt_k = int(cp.ceil(cp.sqrt(k)))
        kmeans1 = cuKMeans(n_clusters=sqrt_k, random_state=0, max_iter=1).fit(X[:, :D//2])
        kmeans2 = cuKMeans(n_clusters=sqrt_k, random_state=0, max_iter=1).fit(X[:, D//2:])
        centroids0, centroids1 = kmeans1.cluster_centers_, kmeans2.cluster_centers_

        seeds = cp.empty((sqrt_k * sqrt_k, D), dtype=cp.float32)
        for i in range(sqrt_k):
            for j in range(sqrt_k):
                row = i * sqrt_k + j
                seeds[row, :D//2] = centroids0[i]
                seeds[row, D//2:] = centroids1[j]
        seeds = seeds[:k]
    elif init not in ['subspaces']:
        raise ValueError("init parameter must be 'subspaces'")

    est = cuKMeans(n_clusters=k, init=seeds, max_iter=max_iter, n_init=1).fit(X)
    if return_sse:
        return est.cluster_centers_, est.labels_, est.inertia_
    return est.cluster_centers_, est.labels_

