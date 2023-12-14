import numpy as np

from scipy.special import logsumexp


class PopulationModel:
    def __init__(self, basis, n_classes):
        self.basis = basis
        self.n_classes = n_classes

        self.class_prob = np.zeros(n_classes)
        self.class_coef = np.zeros((n_classes, basis.dimension))

    def set_class_prob(self, prob):
        self.class_prob[:] = prob

    def set_class_coef(self, coef):
        self.class_coef[:] = coef

    def sample_class_prob(self, rng):
        logits = rng.normal(size=self.n_classes)
        self.class_prob[:] = np.exp(logits - logsumexp(logits))

    def sample_class_coef(self, mean, cov, rng):
        mvn_rvs = rng.multivariate_normal
        self.class_coef[:] = mvn_rvs(mean, cov, size=self.n_classes)

    def sample(self, rng, size=1):
        z = rng.choice(self.n_classes, size=size, p=self.class_prob)
        w = self.class_coef[z]
        return z, w
