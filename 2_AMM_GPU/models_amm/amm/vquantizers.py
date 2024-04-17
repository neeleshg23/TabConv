from __future__ import division, absolute_import

import abc
import matplotlib.pyplot as plt
import cupy as cp
import seaborn as sb

from . import product_quantize as pq
from .kmeans import kmeans
# from old_utils import kmeans 

# ================================================================ misc funcs

def dists_elemwise_dot(x, q):
    return x * q


# XXX: not clear whether this function is correct in general, but
# does always pass the asserts (which capture the invariants we want)
def _insert_zeros(X, nzeros):
    N, D = X.shape
    D_new = D + nzeros
    X_new = cp.zeros((N, D_new), dtype=X.dtype)
    # print("attempting to insert {} zeros into X of shape {}".format(nzeros, X.shape))

    step = int(D / (nzeros + 1)) - 1
    step = max(1, step)
    # print("using step: ", step)

    for i in range(nzeros):
        in_start = step * i
        in_end = in_start + step
        # out_start = in_start + i + 1
        out_start = (step + 1) * i
        out_end = out_start + step
        X_new[:, out_start:out_end] = X[:, in_start:in_end]

    # out_start = out_end
    # out_end += step

    out_end += 1  # account for the last 0
    remaining_len = D - in_end
    out_remaining_len = D_new - out_end
    # print "step", step
    # print "in_start, in_end", in_start, in_end
    # print "out_start, out_end", out_start, out_end
    # print "D, D_new", D, D_new
    # print "remaining_len, out_remaining_len", remaining_len, out_remaining_len
    assert remaining_len == out_remaining_len

    assert remaining_len >= 0
    if remaining_len:
        # X_new[:, out_end:out_end+remaining_len] = X[:, in_end:D]
        X_new[:, out_end:] = X[:, in_end:]

    # print("first cols of old and new X:")
    # print(X[:, 0])
    # print(X_new[:, 0])
    # print(X_new.shape)
    # print((X_new.sum(axis=0) != 0).sum())
    assert X.shape[0] == X_new.shape[0]
    cols_nonzero = X_new.sum(axis=0) != 0
    orig_cols_nonzero = X.sum(axis=0) != 0
    # new_cols_nonzero = cols_nonzero & (~orig_cols_nonzero)
    # print("zero cols: ", np.where(~cols_nonzero)[0])

    assert cols_nonzero.sum() == orig_cols_nonzero.sum()
    nzeros_added = (~cols_nonzero).sum() - (~orig_cols_nonzero).sum()
    assert nzeros_added == nzeros
    # assert np.array_equal(X[:, 0], X_new[:, 0])
    # assert np.array_equal(X[:, -1], X_new[:, -1])

    return X_new


# def ensure_num_cols_multiple_of(X, multiple_of, min_ncols=-1):
def ensure_num_cols_multiple_of(X, multiple_of):
    remainder = X.shape[1] % multiple_of
    if remainder > 0:
        return _insert_zeros(X, multiple_of - remainder)

    return X


# ================================================================ Quantizers


# ------------------------------------------------ Abstract Base Class

class MultiCodebookEncoder(abc.ABC):
    #mark:ncentroids=256
    def __init__(self, ncodebooks, ncentroids=16,
                 quantize_lut=False, upcast_every=-1, accumulate_how='sum'):
        self.ncodebooks = ncodebooks
        self.ncentroids = ncentroids
        self.quantize_lut = quantize_lut
        self.upcast_every = upcast_every if upcast_every >= 1 else 1
        self.upcast_every = min(self.ncodebooks, self.upcast_every)
        assert self.upcast_every in (1, 2, 4, 8, 16, 32, 64, 128, 256)
        self.accumulate_how = accumulate_how

        self.code_bits = int(cp.log2(self.ncentroids))

        # for fast lookups via indexing into flattened array
        self.offsets = (cp.arange(self.ncodebooks, dtype=cp.int32) *
                        self.ncentroids)

    def name(self):
        return "{}_{}x{}b_quantize={}".format(
            self.preproc, self.ncodebooks, self.code_bits,
            int(self.quantize_lut))

    def params(self):
        return {'ncodebooks': self.ncodebooks,
                'code_bits': self.code_bits, 'quantize': self.quantize_lut}


    def dists_enc(self, X_enc, Q_luts, unquantize=False,
                          offset=None, scale=None):
        X_enc = cp.ascontiguousarray(X_enc)

        if unquantize:
            offset = self.total_lut_offset if offset is None else offset
            scale = self.scale_by if scale is None else scale

        all_dists = cp.empty((len(Q_luts), len(X_enc)), dtype=cp.float32)
        for i, lut in enumerate(Q_luts):
            centroid_dists = lut.ravel()[X_enc.ravel()]
            dists = centroid_dists.reshape(X_enc.shape)
            #(batch, n_subspace): (88462, 4)
            if self.upcast_every < 2 or not self.quantize_lut:
                dists = dists.sum(axis=-1)#(88462,)
            else:
                dists = dists.reshape(dists.shape[0], -1, self.upcast_every)
                if self.accumulate_how == 'sum':
                    # sum upcast_every vals, then clip to mirror saturating
                    # unsigned addition, then sum without saturation (like u16)
                    dists = dists.sum(2)
                    dists = cp.clip(dists, 0, 255).sum(axis=-1)
                elif self.accumulate_how == 'mean':
                    # mirror hierarchical avg_epu8
                    # print("reducing using mean!")

                    # print("fraction of low bits that are 1: ",
                    #       np.mean(dists % 2 == 1))  # ya, ~.5, or maybe ~.495

                    while dists.shape[-1] > 2:
                        dists = (dists[:, :, ::2] + dists[:, :, 1::2] + 1) // 2
                    dists = (dists[:, :, 0] + dists[:, :, 1] + 1) // 2
                    dists = dists.sum(axis=-1)  # clipping not needed

                    dists *= self.upcast_every  # convert mean to sum

                    # I honestly don't know why this is the formula, but wow
                    # does it work well
                    bias = self.ncodebooks / 4 * cp.log2(self.upcast_every)
                    dists -= int(bias)

                else:
                    raise ValueError("accumulate_how must be 'sum' or 'mean'")

            if self.quantize_lut and unquantize:
                # dists = (dists / self.scale_by) + self.total_lut_offset
                dists = (dists / scale) + offset
            all_dists[i] = dists

        return all_dists.T

    def dists_enc_cnn(self, X_enc, Q_luts, unquantize=False,
                          offset=None, scale=None):
        X_enc = cp.ascontiguousarray(X_enc)

        if unquantize:
            offset = self.total_lut_offset if offset is None else offset
            scale = self.scale_by if scale is None else scale

        all_dists = cp.empty((len(Q_luts),  len(X_enc)), dtype=cp.float32)

        for i, lut in enumerate(Q_luts):
            centroid_dists = lut.ravel()[X_enc.ravel()]
            dists = centroid_dists.reshape(X_enc.shape)
            dists = dists.sum(axis=-1)
            #dists = np.concatenate((dists))

            if self.quantize_lut and unquantize:
                # dists = (dists / self.scale_by) + self.total_lut_offset
                dists = (dists / scale) + offset
            all_dists[i] = dists

        return all_dists.T.reshape(len(X_enc),-1,len(Q_luts))


# ------------------------------------------------ Product Quantization

def _learn_centroids(X, ncentroids, ncodebooks, subvect_len):
    # print("_learn_centroids(): running kmeans...")
    tot_sse = 0
    
    # ret = np.empty((ncentroids, ncodebooks, subvect_len))
    # X_bar = X - np.mean(X, axis=0)
    # col_sses = np.sum(X_bar * X_bar, axis=0) + 1e-14
    # tot_sse_using_mean = np.sum(col_sses)
    X = cp.asarray(X)
    
    ret = cp.empty((ncentroids, ncodebooks, subvect_len))
    tot_sse = 0

    X_bar = X - cp.mean(X, axis=0)
    col_sses = cp.sum(X_bar * X_bar, axis=0) + 1e-14
    tot_sse_using_mean = cp.sum(col_sses)
    
    for i in range(ncodebooks):
        print("running kmeans in subspace {}/{}...".format(
            i + 1, ncodebooks), end=" ")
        start_col = i * subvect_len
        end_col = start_col + subvect_len
        X_in = X[:, start_col:end_col]
        # centroids, labels = kmeans(X_in, ncentroids)
        centroids, labels, sse = kmeans(X_in, ncentroids, return_sse=True)

        # X_bar = X_in - np.mean(X_in, axis=0)
        # sse_using_mean = np.sum(X_bar * X_bar) + 1e-14
        subspace_sse = cp.sum(col_sses[start_col:end_col])
        print("mse / {{var(X_subs), var(X)}}: {:.3g}, {:.3g}".format(
            sse / subspace_sse, sse * ncodebooks / tot_sse_using_mean))
        tot_sse += sse
        # print("centroids shape: ", centroids.shape)
        # print("ret shape: ", ret.shape)
        ret[:, i, :] = centroids
        # ret[:, i, :] = centroids.get()

    print("--- total mse / var(X): {:.3g}".format(tot_sse / tot_sse_using_mean))

    return ret


def _fit_pq_lut(q, centroids, elemwise_dist_func):
    _, ncodebooks, subvect_len = centroids.shape
    q = q.reshape((1, ncodebooks, subvect_len))
    q_dists = cp.sum(centroids * q, axis=-1) #dists: distribution?
    #q_dists: (16,2): ncentroids row, ncodebooks column table
    return q_dists  # ncentroids, ncodebooks, row-major


class PQEncoder(MultiCodebookEncoder):

    #mark:ncentroids=256
    def __init__(self, ncodebooks, ncentroids=16,
                 elemwise_dist_func=dists_elemwise_dot,
                 preproc='PQ', encode_algo=None, quantize_lut=False,
                 upcast_every=-1, accumulate_how='sum',
                 **preproc_kwargs):
        super().__init__(
            ncodebooks=ncodebooks, ncentroids=ncentroids,
            quantize_lut=quantize_lut, upcast_every=upcast_every,
            accumulate_how=accumulate_how)
        self.elemwise_dist_func = elemwise_dist_func
        self.preproc = preproc
        self.encode_algo = encode_algo
        self.preproc_kwargs = preproc_kwargs

    def _pad_ncols(self, X):
        return ensure_num_cols_multiple_of(X, self.ncodebooks)

    def fit(self, X, Q=None):#kmeans learn centroids
        self.subvect_len = int(cp.ceil(X.shape[1] / self.ncodebooks))
        X = self._pad_ncols(X)
        self.centroids = None
        if self.centroids is None:
            self.centroids = _learn_centroids(
                X, self.ncentroids, self.ncodebooks, self.subvect_len)

    def name(self):
        return "{}_{}".format(self.preproc, super().name())

    def params(self):
        d = super().params()
        d['_preproc'] = self.preproc
        return d

    def encode_Q(self, Q, quantize=True):#weight * centoids -> lut
        # quantize param enables quantization if set in init; separate since
        # quantization learning needs to call this func, but vars like
        # lut_offsets aren't set when this function calls it

        Q = cp.atleast_2d(Q)
        Q = self._pad_ncols(Q)

        luts = cp.zeros((Q.shape[0], self.ncodebooks, self.ncentroids))
        # print("Q shape: ", Q.shape) Q(64, 100)
        for i, q in enumerate(Q):
            lut = _fit_pq_lut(q, centroids=self.centroids,
                              elemwise_dist_func=self.elemwise_dist_func)
            if self.quantize_lut and quantize:
                lut = cp.maximum(0, lut - self.lut_offsets)
                lut = cp.floor(lut * self.scale_by).astype(cp.int32)
                lut = cp.minimum(lut, 255)
            luts[i] = lut.T
        return luts

    def encode_X(self, X, **sink):#x->closest centroids->indexes
        X = self._pad_ncols(X)

        idxs = pq._encode_X_pq(X, codebooks=self.centroids)

        return idxs + self.offsets  # offsets let us index into raveled dists


## CNN

def _fit_pq_lut_cnn(q, centroids):
    _, ncodebooks, subvect_len = centroids.shape
    q = q.reshape((1, ncodebooks, subvect_len))
    q_dists = cp.sum(centroids * q, axis=-1) #dists: distribution?
    #q_dists: (16,2): ncentroids row, ncodebooks column table
    return q_dists  # ncentroids, ncodebooks, row-major



class PQEncoder_CNN(MultiCodebookEncoder):

    #mark:ncentroids=256
    def __init__(self, ncodebooks, ncentroids=16,
                 elemwise_dist_func=dists_elemwise_dot,
                 preproc='PQ', encode_algo=None, quantize_lut=False,
                 upcast_every=-1, accumulate_how='sum',
                 **preproc_kwargs):
        super().__init__(
            ncodebooks=ncodebooks, ncentroids=ncentroids,
            quantize_lut=quantize_lut, upcast_every=upcast_every,
            accumulate_how=accumulate_how)
        self.elemwise_dist_func = elemwise_dist_func
        self.preproc = preproc
        self.encode_algo = encode_algo
        self.preproc_kwargs = preproc_kwargs

    def _pad_ncols(self, X):
        return ensure_num_cols_multiple_of(X, self.ncodebooks)

    def fit(self, X, Q=None):#kmeans learn centroids
        self.subvect_len = int(cp.ceil(X.shape[1] / self.ncodebooks))
        X = self._pad_ncols(X)
        self.centroids = None
        if self.centroids is None:
            self.centroids = _learn_centroids(
                X, self.ncentroids, self.ncodebooks, self.subvect_len)

    def name(self):
        return "{}_{}".format(self.preproc, super().name())

    def params(self):
        d = super().params()
        d['_preproc'] = self.preproc
        return d

    def encode_Q(self, Q, quantize=True):#weight * centoids -> lut
        # quantize param enables quantization if set in init; separate since
        # quantization learning needs to call this func, but vars like
        # lut_offsets aren't set when this function calls it

        #Q = np.repeat(Q, self.ncodebooks, axis=0).transpose() #Q:(9,4).

        Q = cp.atleast_2d(Q)
        Q = self._pad_ncols(Q)
        luts = cp.zeros((Q.shape[0], self.ncodebooks, self.ncentroids))
        # print("Q shape: ", Q.shape) Q(900,4)
        for i, q in enumerate(Q):
            lut = _fit_pq_lut_cnn(q, centroids=self.centroids)
            if self.quantize_lut and quantize:
                lut = cp.maximum(0, lut - self.lut_offsets)
                lut = cp.floor(lut * self.scale_by).astype(cp.int32)
                lut = cp.minimum(lut, 255)
            luts[i] = lut.T
            #(4, 100, 16)
        return luts

    def encode_X(self, X, **sink):#x->closest centroids->indexes
        X = self._pad_ncols(X)

        idxs = pq._encode_X_pq(X, codebooks=self.centroids)

        return idxs + self.offsets  # offsets let us index into raveled dists



def main():
    X = cp.ones((3, 75), dtype=cp.int32)
    _insert_zeros(X, 53)


if __name__ == '__main__':
    main()
