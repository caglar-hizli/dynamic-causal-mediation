import logging

import autograd
import autograd.numpy as np
import pandas as pd

from autograd.scipy.stats import multivariate_normal as mvn
from scipy.optimize import minimize

# from ptk import scg
from models.benchmarks.schulam.autodiff import chol_to_cov
from models.benchmarks.schulam.bsplines import BSplines
from models.benchmarks.schulam.autodiff import packing_funcs


class LMMGP:
    def __init__(self, low, high, n_bases, degree):
        self.kernel = make_kernel(low, high, n_bases, degree)
        self.params = {}
        self.params['cov_chol'] = np.eye(n_bases).ravel()
        self.params['ln_noise'] = np.zeros(1)
        self.params['m'] = np.zeros(1)

    def predict(self, x_star, y, x):
        n_star = len(x_star[0])
        prior_mean = self.params['m'] * np.ones(n_star)

        if len(y) == 0:
            return prior_mean

        k_star = self.kernel(self.params, x_star)
        k_obs = self.kernel(self.params, x)
        k_cross = self.kernel(self.params, x_star, x)

        obs_mean = self.params['m'] * np.ones(len(y))
        mean = prior_mean + np.dot(k_cross, np.linalg.solve(k_obs, y - obs_mean))

        return mean

    def fit(self, samples):
        pack, unpack = packing_funcs(self.params)
        
        def obj(w):
            p = unpack(w)
            ll = 0.0
        
            for y, x in samples:
                mean = p['m'] * np.ones(len(y))
                cov = self.kernel(p, x)
                ll += mvn.logpdf(y, mean, cov)

            return -ll

        grad = autograd.grad(obj)

        def callback(i, f, g, x):
            logging.info('obj={:.4f}'.format(f))

        # solution = scg.scg(obj, pack(self.params), grad, callback=callback)
        solution = minimize(obj, pack(self.params), jac=grad, callback=callback, method='BFGS')
        self.params = unpack(solution['x'])


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


def make_kernel(low, high, n_bases, degree):
    basis = BSplines(low, high, n_bases, degree, boundaries='space')

    def kernel(params, x1, x2=None):
        bspline_cov = chol_to_cov(params['cov_chol'], n_bases, False)
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
            gram += noise_var * np.eye(len(t1))

        return gram

    return kernel
