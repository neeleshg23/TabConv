import numpy as np 
# ================================================================ metrics

def _cossim(Y, Y_hat):
    ynorm = np.linalg.norm(Y) + 1e-20
    yhat_norm = np.linalg.norm(Y_hat) + 1e-20
    return ((Y / ynorm) * (Y_hat / yhat_norm)).sum()

def layer_cossim(layer_exact, layer_amm):
    res = []
    n = len(layer_exact)
    for i in range(n):
        res.append(_cossim(layer_exact[i], layer_amm[i]))
    return res