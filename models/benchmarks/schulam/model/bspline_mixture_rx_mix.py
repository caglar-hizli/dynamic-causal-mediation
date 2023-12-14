import logging

import autograd
import autograd.numpy as np

from autograd.scipy.misc import logsumexp
from scipy.optimize import minimize

from models.benchmarks.schulam.autodiff import vec_mvn_logpdf
from models.benchmarks.schulam.longitudinal.util import cluster_trajectories
from models.benchmarks.schulam.autodiff import packing_funcs


def model_selection(basis, n_class_options, rx_w, samples):
    models = []

    for k in n_class_options:
        m = TreatmentBSplineMixture(basis, k, rx_w)
        m.fit(samples)
        models.append((m.bic(samples), m))

    return models


class TreatmentBSplineMixture:
    def __init__(self, basis, n_classes, rx_w):
        self.basis = basis
        self.n_classes = n_classes
        self.rx_w = rx_w

        self.params = {}
        self.params['prob_logit'] = np.zeros(n_classes)
        self.params['class_coef'] = np.zeros((n_classes, basis.dimension))
        self.params['ln_cov_w'] = np.zeros(1)
        self.params['ln_cov_y'] = np.zeros(1)
        self.params['treatment'] = np.zeros(2)

    def predict(self, x_star, y, x):
        p_z = self.class_posterior(y, x)

        n_classes = 2 * self.n_classes  # Each subtype has treat/no treat variant.
        p_y = [self.class_predictive(x_star, y, x, z) for z in range(n_classes)]

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
            f += np.sum(p['treatment']**2)

            return f

        grad = autograd.grad(obj)

        def callback(w):
            p = unpack(w)
            v_w = np.exp(p['ln_cov_w'])
            v_y = np.exp(p['ln_cov_y'])
            rx = p['treatment'][0]
            logging.info('obj={:.4f}, v_w={:.4f}, v_y={:.4f}, rx={:.2f}'.format(obj(w), v_w[0], v_y[0], rx))

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
        s = 'TreatmentBSplineMixture(n_classes={})'.format(self.n_classes)
        return s


def log_likelihood(params, basis, w, y, x):
    ln_joint = joint_ln_probability(params, basis, w, y, x)
    return logsumexp(ln_joint)


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
    b = basis.design(t)
    m1 = np.dot(params['class_coef'], b.T)
    m2 = params['treatment'][0] * rx_basis((t, rx), w)
    return np.vstack((m1, m1 + m2))


def rx_basis(x, w):
    t, rx = x
    t_rx = t[rx == 1]
    d = t[:, None] - t_rx[None, :]
    treated = (d > 0) & (d <= w)
    return np.sum(treated, axis=1)


def cov_fn(params, basis, t1, t2=None, eps=1e-3):
    v_w = np.exp(params['ln_cov_w'])
    v_y = np.exp(params['ln_cov_y'])

    if t2 is None:
        t2 = t1
        symmetric = True

    else:
        symmetric = False

    b1 = basis.design(t1)
    b2 = basis.design(t2)
    cov = v_w * np.dot(b1, b2.T)

    if symmetric:
        cov += (v_y + eps) * np.eye(len(t1))

    return cov


def ln_prior(params):
    z_logits = params['prob_logit']
    ln_p_z = z_logits - logsumexp(z_logits)

    rx_logit = params['treatment'][1]
    p_rx = 1.0 / (1.0 + np.exp(-rx_logit))
    ln_p_rx = np.log(np.array([1 - p_rx, p_rx]))

    return (ln_p_rx[:, None] + ln_p_z[None, :]).ravel()
