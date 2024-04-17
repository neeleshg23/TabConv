import cupy as cp
# ================================================================ metrics

def _cossim(Y, Y_hat):
    ynorm = cp.linalg.norm(Y) + 1e-20
    yhat_norm = cp.linalg.norm(Y_hat) + 1e-20
    return ((Y / ynorm) * (Y_hat / yhat_norm)).sum()

def layer_cossim(layer_exact, layer_amm):
    res = []
    n = len(layer_exact)
    for i in range(n):
        res.append(_cossim(layer_exact[i], layer_amm[i]))
    return res