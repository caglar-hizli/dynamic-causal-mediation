import numpy as np

from scipy.cluster.hierarchy import linkage, cut_tree

from . import lmm


def cluster_trajectories(trajectories, n_clusters, basis, method='complete'):
    coef = estimate_coef(trajectories, basis)
    link = linkage(coef, method)
    clusters = cut_tree(link, n_clusters).ravel()
    return coef, clusters


def estimate_coef(trajectories, basis):
    data = [(y, _intercept(t), basis.design(t)) for y, t in trajectories]
    model = lmm.learn_lmm(data)
    coef = [model.posterior(*x)[0] for x in data]
    return np.array(coef)


def _intercept(x):
    n = len(x)
    return np.ones(n)[:, None]
