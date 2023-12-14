import os.path

import numpy as np


class OutcomeModelFpca:
    def __init__(self, eigen_basis, beta, aug_x, theta):
        self.eigen_basis = np.transpose(eigen_basis, (0, 2, 1))  # (M, n_outcome, K=6)
        self.beta = beta  # (M, n_patient, K=6)
        n_patient = self.beta.shape[1]
        self.theta = theta  # (M, 2)
        self.X_train = np.vstack([aug_x, aug_x[-1, :]])  # (n_patient*n_outcome, 2)
        self.X_train = self.X_train.reshape((n_patient, -1, 2))  # (n_patient, n_outcome, 2)
        self.X_train = np.transpose(self.X_train, (2, 0, 1))  # (2, n_patient, n_outcome)
        self.eigen = self.beta @ self.eigen_basis  # (M, n_patient, n_outcome)
        intercept = np.einsum('mi,ikl->mkl', self.theta[:, 0:1], self.X_train[0:1, :, :])  # (M, n_patient, n_outcome)
        self.baseline = self.eigen + intercept  # (M, n_patient, n_outcome)

    def predict_outcome(self, x, dosage, pidx):
        X = np.stack([np.ones_like(dosage), dosage])  # (2, n_outcome)
        mean = self.eigen[:, pidx, :] + self.theta @ X  # (M, n_outcome)
        return mean.mean(0)  # (n_outcome,)

    def predict_baseline(self, x, pidx):
        return self.baseline[:, pidx, :].mean(0)  # (n_outcome,)


def get_dosage_fpca(x, actions, args):
    t, m = actions[:, 0], actions[:, 1]
    mediators = []
    for xi in x:
        mi = 0.0
        effective_mask = np.logical_and(t < xi, np.abs(xi - t) < args.T_treatment)
        mi += np.sum(m[effective_mask])
        mediators.append(mi)
    return np.array(mediators)


def load_outcome_params_fpca(path):
    mcmc = np.load(path, allow_pickle=True)
    eigen_basis = mcmc['eigen']
    beta = mcmc['beta']
    aug_x = mcmc['aug_x']
    theta = mcmc['theta']
    # xx = np.linspace(0, 50, 100)
    # gamma_samples = np.stack(
    #     [np.exp(mu[i - 1]) + alpha[i - 1, 0] * gamma.pdf(xx, a=a[i - 1], scale=1 / rate[i - 1]) for i in
    #      thinned_iters[-10:]])
    # plt.plot(xx, gamma_samples.T)
    # plt.show()
    # plt.plot(xx, gamma_samples.mean(0), 'b-')
    # plt.fill_between(xx,
    #                  gamma_samples.mean(0) - 1.96 * np.sqrt(gamma_samples.var(0)),
    #                  gamma_samples.mean(0) + 1.96 * np.sqrt(gamma_samples.var(0)),
    #                  color='blue',
    #                  alpha=0.2,
    #                  )
    # plt.show()
    return eigen_basis, beta, aug_x, theta
