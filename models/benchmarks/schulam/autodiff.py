import autograd.numpy as np

from autograd.scipy.stats import multivariate_normal as mvn
from autograd.scipy.special import logsumexp


def packing_funcs(params):
    keys = sorted(params.keys())
    sizes = {k:params[k].size for k in keys}
    shapes = {k:params[k].shape for k in keys}

    def pack(params):
        flat = [params[k].ravel() for k in keys]
        return np.concatenate(flat)

    def unpack(packed):
        params = {}
        num_read = 0

        for k in keys:
            n = sizes[k]
            s = shapes[k]
            params[k] = packed[num_read:(num_read + n)].reshape(s)
            num_read += n

        return params

    return pack, unpack


def chol_to_cov(chol, d, diag):
    lower = np.tril(chol.reshape((d, d)))
    if diag:
        lower = np.diag(np.diag(lower))
    return np.dot(lower, lower.T)


def kronecker(a, b):
    n_a, m_a = a.shape
    n_b, m_b = b.shape

    rx = np.repeat(np.arange(n_a), n_b)[:, None]
    cx = np.repeat(np.arange(m_a), m_b)[None, :]

    return a[rx, cx] * np.tile(b, (n_a, m_a))


def vec_mvn_logpdf(x, m_vec, cov):
    m_vec = np.atleast_2d(m_vec)
    logdet = np.linalg.slogdet(cov)[1]
    d = m_vec.shape[1]
    r = x - m_vec
    q = np.sum(np.linalg.solve(cov, r.T).T * r, axis=1)
    return -0.5 * (d * np.log(2 * np.pi) + logdet + q)
