import logging

import autograd
import autograd.numpy as np

from autograd.scipy.special import logsumexp
from numpy.linalg.linalg import LinAlgError
from scipy.optimize import minimize

from ..autodiff import packing_funcs, vec_mvn_logpdf
from ..longitudinal.util import cluster_trajectories


def model_selection(basis, n_class_options, samples):
    models = {}

    for k in n_class_options:
        m = BSplineMixture(basis, k)
        m.fit(samples)
        models[k] = (m.bic(samples), m)

    return models


class BSplineMixture:
    def __init__(self, basis, n_classes):
        self.basis = basis
        self.n_classes = n_classes

        self.params = {}
        self.params['prob_logit'] = np.zeros(n_classes)
        self.params['class_coef'] = np.zeros((n_classes, basis.dimension))
        self.params['ln_cov_y'] = np.zeros(1)

    def predict(self, x_star, y, x):
        p_z = self.class_posterior(y, x)
        p_y = [self.class_predictive(x_star, y, x, z) for z in range(self.n_classes)]
        return sum(w * y_hat for w, (y_hat, _) in zip(p_z, p_y))

    def class_posterior(self, y, x):
        return class_posterior(self.params, self.basis, y, x)

    def class_predictive(self, x_star, y, x, z):
        return class_predictive(self.params, self.basis, x_star, y, x, z)

    def fit(self, samples):
        self._initialize(samples)

        pack, unpack = packing_funcs(self.params)

        def obj(w):
            p = unpack(w)
            f = 0.0

            for y, x in samples:
                f -= log_likelihood(p, self.basis, y, x)

            f += np.sum(p['prob_logit']**2)
            f += np.sum(p['class_coef']**2)            

            return f

        grad = autograd.grad(obj)

        def callback(w):
            p = unpack(w)
            logging.info('obj={:.4f}'.format(obj(w)))

        solution = minimize(obj, pack(self.params), jac=grad, method='BFGS', callback=callback)
        self.params = unpack(solution['x'])

    def bic(self, samples):
        p, b = self.params, self.basis
        ll = sum(log_likelihood(p, b, y, x) for y, x in samples)

        pack, _ = packing_funcs(p)
        num_params = len(pack(p))

        return -2 * ll + num_params * np.log(len(samples))

    def _initialize(self, samples):
        trajectories = [(y, t) for y, (t, rx) in samples]
        coef, clusters = cluster_trajectories(
            trajectories, self.n_classes, self.basis)

        for k in range(self.n_classes):
            w = coef[clusters == k].mean(axis=0)
            self.params['class_coef'][k] = w

    def __repr__(self):
        s = 'BSplineMixture(n_classes={})'.format(self.n_classes)
        return s


def log_likelihood(params, basis, y, x):
    ln_joint = joint_ln_probability(params, basis, y, x)
    return logsumexp(ln_joint)


def class_posterior(params, basis, y, x):
    if len(y) == 0:
        return np.exp(ln_prior(params))

    ln_joint = joint_ln_probability(params, basis, y, x)
    return np.exp(ln_joint - logsumexp(ln_joint))


def class_predictive(params, basis, x_star, y, x, z):
    t_star, rx_star = x_star
    prior_mean = mean_fn(params, basis, t_star)[z]
    prior_cov = cov_fn(params, basis, t_star)

    if len(y) == 0:
        return prior_mean, prior_cov

    t, rx = x
    obs_mean = mean_fn(params, basis, t)[z]
    obs_cov = cov_fn(params, basis, t)

    cross_cov = cov_fn(params, basis, t_star, t)

    alpha = np.linalg.solve(obs_cov, cross_cov.T).T
    mean = prior_mean + np.dot(alpha, y - obs_mean)
    cov = prior_cov - np.dot(alpha, cross_cov.T)

    return mean, cov


def joint_ln_probability(params, basis, y, x):
    t, rx = x
    m = mean_fn(params, basis, t)
    c = cov_fn(params, basis, t)

    ln_p_z = ln_prior(params)
    ln_p_y = vec_mvn_logpdf(y, m, c)

    return ln_p_z + ln_p_y


def mean_fn(params, basis, t):
    w_z = params['class_coef']
    b = basis.design(t)
    return np.dot(w_z, b.T)


def cov_fn(params, basis, t1, t2=None, eps=1e-3):
    v_y = np.exp(params['ln_cov_y'])
    symmetric = t2 is None

    if symmetric:
        cov = (v_y + eps) * np.eye(len(t1))
    else:
        cov = np.zeros((len(t1), len(t2)))

    return cov


def ln_prior(params):
    logits = params['prob_logit']
    return logits - logsumexp(logits)
