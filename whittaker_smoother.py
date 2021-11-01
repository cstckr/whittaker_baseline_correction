from scipy import sparse
from scipy.sparse.linalg import splu

def whittaker(y, d=1, lmbda=1, **kwargs):
    w = kwargs.get("w", np.ones(len(y)))
    m = len(y)
    D = sparse.eye(m, format="csc")
    for i in range(d): D = D[1:] - D[:-1]
    W = sparse.spdiags(w, 0, m, m, format="csc")
    z = splu(W + lmbda * D.T * D).solve((y * w))
    return z
