import logging

import autograd
import autograd.numpy as np

from autograd.scipy.special import logsumexp
from numpy.linalg.linalg import LinAlgError
from scipy.optimize import minimize

from ..autodiff import packing_funcs, vec_mvn_logpdf
from ..longitudinal.util import cluster_trajectories


def model_selection(basis, n_class_options, rx_w_options, samples):
    models = {}

    for k in n_class_options:
        for w in rx_w_options:
            m = TreatmentBSplineMixture(basis, k, w)
            m.fit(samples)
            models[(k, w)] = (m.bic(samples), m)

    return models


class TreatmentBSplineMixture:
    def __init__(self, basis, n_classes, rx_w):
        self.basis = basis
        self.n_classes = n_classes
        self.rx_w = rx_w

        self.params = {}
        self.params['prob_logit'] = np.zeros(n_classes)
        self.params['class_coef'] = np.zeros((n_classes, basis.dimension))
        self.params['ln_ou'] = np.zeros(2)
        self.params['ln_cov_y'] = np.zeros(1)
        self.params['treatment'] = np.zeros(1)

    def predict(self, x_star, y, x):
        p_z = self.class_posterior(y, x)
        p_y = [self.class_predictive(x_star, y, x, z) for z in range(self.n_classes)]
        return sum(w * y_hat for w, (y_hat, _) in zip(p_z, p_y))

    def class_posterior(self, y, x):
        return class_posterior(self.params, self.basis, self.rx_w, y, x)

    def class_predictive(self, x_star, y, x, z):
        return class_predictive(self.params, self.basis, self.rx_w, x_star, y, x, z)

    def fit(self, samples):
        self._initialize(samples)

        pack, unpack = packing_funcs(self.params)

        def obj(w):
            p = unpack(w)
            f = 0.0
            for y, x in samples:
                ll = log_likelihood(p, self.basis, self.rx_w, y, x)
                f -= ll
            f += np.sum(p['prob_logit']**2)
            f += np.sum(p['class_coef']**2)
            # f += np.sum(p['ln_ou']**2)
            return f

        grad = autograd.grad(obj)

        def callback(w):
            p = unpack(w)
            logging.info('obj={:.4f}'.format(obj(w)))

        solution = minimize(obj, pack(self.params), jac=grad, method='BFGS', callback=callback)
        self.params = unpack(solution['x'])

    def bic(self, samples):
        p, b, w = self.params, self.basis, self.rx_w
        ll = sum(log_likelihood(p, b, w, y, x) for y, x in samples)

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
        s = 'TreatmentBSplineMixture(n_classes={}, rx_w={:.1f})'.format(self.n_classes, self.rx_w)
        return s


def log_likelihood(params, basis, w, y, x):
    ln_likel = logsumexp(joint_ln_probability(params, basis, w, y, x))
    return ln_likel


def class_posterior(params, basis, w, y, x):
    if len(y) == 0:
        return np.exp(ln_prior(params))

    ln_joint = joint_ln_probability(params, basis, w, y, x)
    return np.exp(ln_joint - logsumexp(ln_joint))


def class_predictive(params, basis, w, x_star, y, x, z):
    t_star, rx_star = x_star
    prior_mean = mean_fn(params, basis, w, t_star, rx_star)[z]
    prior_cov = cov_fn(params, basis, t_star)

    if len(y) == 0:
        return prior_mean, prior_cov

    t, rx = x
    obs_mean = mean_fn(params, basis, w, t, rx)[z]
    obs_cov = cov_fn(params, basis, t)

    cross_cov = cov_fn(params, basis, t_star, t)

    alpha = np.linalg.solve(obs_cov, cross_cov.T).T
    mean = prior_mean + np.dot(alpha, y - obs_mean)
    cov = prior_cov - np.dot(alpha, cross_cov.T)

    return mean, cov


def joint_ln_probability(params, basis, w, y, x):
    t, rx = x
    m = mean_fn(params, basis, w, t, rx)
    c = cov_fn(params, basis, t)

    ln_p_z = ln_prior(params)
    ln_p_y = vec_mvn_logpdf(y, m, c)

    return ln_p_z + ln_p_y


def mean_fn(params, basis, w, t, rx):
    w_z = params['class_coef']
    b = basis.design(t)
    m1 = np.dot(w_z, b.T)
    m2 = params['treatment'] * rx_basis((t, rx), w)
    return m1 + m2


def rx_basis(x, w, stack=False):
    t, rx = x
    t_rx = t[rx == 1]
    d = t[:, None] - t_rx[None, :]
    treated = (d > 0) & (d <= w)

    if stack:
        return np.sum(treated, axis=1)
    else:
        return np.any(treated, axis=1).astype(float)


def cov_fn(params, basis, t1, t2=None, eps=1e-2):
    ln_a, ln_l = params['ln_ou']
    v_y = np.exp(params['ln_cov_y'])
    symmetric = t2 is None

    if symmetric:
        # cov = _matern_cov(t1, t1, ln_a, ln_l) + (v_y + eps) * np.eye(len(t1))
        cov = _ou_cov(t1, t1, ln_a, ln_l) + (v_y + eps) * np.eye(len(t1))
        # cov = (v_y + eps) * np.eye(len(t1))
    else:
        # cov = _matern_cov(t1, t2, ln_a, ln_l)
        cov = _ou_cov(t1, t2, ln_a, ln_l)
        # cov = np.zeros((len(t1), len(t2)))

    return cov


def _ou_cov(t1, t2, ln_a, ln_l):
    a = np.exp(ln_a)
    l = np.exp(ln_l)
    D = np.expand_dims(t1, 1) - np.expand_dims(t2, 0)
    return a * np.exp(-np.abs(D) / l)


def _matern_cov(t1, t2, ln_a, ln_l):
    a = np.exp(ln_a)
    l = np.exp(ln_l)
    D = np.abs(np.expand_dims(t1, 1) - np.expand_dims(t2, 0)) / l
    return a * (1 + np.sqrt(3) * D) * np.exp(-np.sqrt(3) * D)


def ln_prior(params):
    logits = params['prob_logit']
    return logits - logsumexp(logits)
