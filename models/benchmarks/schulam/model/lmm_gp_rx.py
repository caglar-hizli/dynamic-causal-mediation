import logging

import autograd
import autograd.numpy as np
import pandas as pd

from autograd.scipy.stats import multivariate_normal as mvn
from scipy.optimize import minimize

from models.benchmarks.schulam.autodiff import chol_to_cov
from models.benchmarks.schulam.bsplines import BSplines
from models.benchmarks.schulam.autodiff import packing_funcs


class TreatmentLMMGP:
    def __init__(self, low, high, n_bases, degree, rx_w):
        self.kernel = make_kernel(low, high, n_bases, degree)
        self.rx_w = rx_w
        self.params = {}
        self.params['cov_chol'] = np.eye(n_bases).ravel()
        self.params['ln_noise'] = np.zeros(1)
        self.params['b'] = np.zeros(1)

    def predict(self, x_star, y, x):
        prior_mean = self.params['b'] * rx_basis(x_star, self.rx_w, x)

        if len(y) == 0:
            return prior_mean

        k_star = self.kernel(self.params, x_star)
        k_obs = self.kernel(self.params, x)
        k_cross = self.kernel(self.params, x_star, x)

        obs_mean = self.params['b'] * rx_basis(x, self.rx_w)
        mean = prior_mean + np.dot(k_cross, np.linalg.solve(k_obs, y - obs_mean))

        return mean

    def fit(self, samples):
        pack, unpack = packing_funcs(self.params)

        def obj(w):
            p = unpack(w)
            ll = 0.0

            for y, x in samples:
                mean = p['b'] * rx_basis(x, self.rx_w)
                cov = self.kernel(p, x)
                ll += mvn.logpdf(y, mean, cov)

            return -ll

        grad = autograd.grad(obj)

        def cb(w):
            m = 'obj={:.4f}'.format(obj(w))
            logging.info(m)

        sol = minimize(obj, pack(self.params), jac=grad, callback=cb, method='BFGS')
        self.params = unpack(sol['x'])

    def fit_treatment(self, samples):
        pack, unpack = packing_funcs(self.params)
        params = self.params

        def obj(w):
            p = unpack(w)
            p['cov_chol'] = params['cov_chol']
            p['ln_noise'] = params['ln_noise']
            ll = 0.0

            for y, x in samples:
                mean = p['b'] * rx_basis(x, self.rx_w)
                cov = self.kernel(p, x)
                ll += mvn.logpdf(y, mean, cov)

            return -ll

        grad = autograd.grad(obj)

        def cb(w):
            p = unpack(w)
            m = 'obj={:.4f}, rx={:.4f}'.format(obj(w), p['b'][0])
            logging.info(m)

        sol = minimize(obj, pack(self.params), jac=grad, callback=cb, method='BFGS')
        self.params = unpack(sol['x'])

    def ln_likel(self, samples):
        p = self.params
        ll = 0.0

        for y, x in samples:
            mean = p['b'] * rx_basis(x, self.rx_w)
            cov = self.kernel(p, x)
            ll += mvn.logpdf(y, mean, cov)

        return ll


class TreatmentRegression:
    def __init__(self, rx_w):
        self.rx_w = rx_w
        self.params = {}
        self.params['ln_noise'] = np.zeros(1)
        self.params['b'] = np.zeros(1)

    # def predict(self, x_star, y, x):
    #     prior_mean = self.params['b'] * rx_basis(x_star, self.rx_w, x)

    #     if len(y) == 0:
    #         return prior_mean

    #     k_star = self.kernel(self.params, x_star)
    #     k_obs = self.kernel(self.params, x)
    #     k_cross = self.kernel(self.params, x_star, x)

    #     obs_mean = self.params['b'] * rx_basis(x, self.rx_w)
    #     mean = prior_mean + np.dot(k_cross, np.linalg.solve(k_obs, y - obs_mean))

    #     return mean

    def fit(self, samples):
        pack, unpack = packing_funcs(self.params)

        def obj(w):
            p = unpack(w)
            ll = 0.0

            for y, x in samples:
                mean = p['b'] * rx_basis(x, self.rx_w)
                cov = (np.exp(p['ln_noise']) + 1e-3) * np.eye(len(y))
                ll += mvn.logpdf(y, mean, cov)

            return -ll

        grad = autograd.grad(obj)

        def cb(w):
            m = 'obj={:.4f}'.format(obj(w))
            logging.info(m)

        sol = minimize(obj, pack(self.params), jac=grad, callback=cb, method='BFGS')
        self.params = unpack(sol['x'])

    def ln_likel(self, samples):
        p = self.params
        ll = 0.0

        for y, x in samples:
            mean = p['b'][0] + p['b'][1] * rx_basis(x, self.rx_w)
            cov = (np.exp(p['ln_noise']) + 1e-3) * np.eye(len(y))
            ll += mvn.logpdf(y, mean, cov)        

        return ll


def evaluate(model, samples, pred_times):
    results = []

    def xi(x, i):
        return x[0][i], x[1][i]

    for t0 in pred_times:
        for i, (y, x) in enumerate(samples):
            t, rx = x
            obs = t <= t0

            if sum(obs) == len(y):
                continue

            y_hat = model.predict(x, y[obs], xi(x, obs))

            results.append(pd.DataFrame(
                np.c_[
                    np.repeat(i, len(y)),
                    np.repeat(t0, len(y)),
                    obs,
                    t, y, y_hat
                ],
                columns = [
                    'sample_num',
                    'pred_time',
                    'observed',
                    't', 'y', 'y_hat'
                ]
            ))

    return pd.concat(results, axis=0)


def make_kernel(low, high, n_bases, degree, diag=False):
    basis = BSplines(low, high, n_bases, degree, boundaries='space')

    def kernel(params, x1, x2=None):
        bspline_cov = chol_to_cov(params['cov_chol'], n_bases, diag)
        noise_var = np.exp(params['ln_noise'])
        
        if x2 is None:
            symmetric = True
            x2 = x1
        else:
            symmetric = False

        t1 = x1[0]
        t2 = x2[0]

        b1 = basis.design(t1)
        b2 = basis.design(t2)

        gram = np.dot(b1, np.dot(bspline_cov, b2.T))

        if symmetric:
            gram += (noise_var + 1e-3) * np.eye(len(t1))

        return gram

    return kernel


def rx_basis(x, w):
    t, rx = x
    rx_t = t[rx == 1]
    d = t[:, None] - rx_t[None, :]
    treated = (d > 0) & (d <= w)
    return np.sum(treated, axis=1)
