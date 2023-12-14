import gpflow
import numpy as np
import tensorflow as tf
from gpflow import Parameter
from gpflow import kullback_leiblers
from gpflow.config import default_float
from gpflow.utilities import positive, triangular, to_default_float
from gpflow.conditionals import conditional
from scipy.stats import ncx2

from models.mediator.gprpp.kernel import get_hyperparameters, get_mark_output, get_mark_lengthscales
from models.mediator.gprpp.utils.Gtilde import tf_Gtilde_lookup
from models.mediator.gprpp.utils.phivector import tf_calc_Phi_SqExp_vec
from models.mediator.gprpp.utils.psimatrix import tf_calc_Psi_matrix_vec
from models.mediator.gprpp.utils.tf_utils import tf_vec_mat_vec_mul, tf_vec_dot


def _integrate_log_fn_sqr(mean, var):
    """
    ∫ log(f²) N(f; μ, σ²) df  from -∞ to ∞
    """
    z = -0.5 * tf.square(mean) / var
    C = 0.57721566  # Euler-Mascheroni constant
    G = tf_Gtilde_lookup(z)
    return -G + tf.math.log(0.5 * var) - C


def integrate_log_fn_sqr(mean, var):
    # N = tf_len(μn)
    integrated = _integrate_log_fn_sqr(mean, var)
    point_eval = tf.math.log(mean ** 2)  # TODO use mvnquad instead?
    # TODO explain
    return tf.where(tf.math.is_nan(integrated), point_eval, integrated)


class GPRPP(gpflow.models.GPModel, gpflow.models.ExternalDataTrainingLossMixin):
    def __init__(
        self,
        inducing_variable: gpflow.inducing_variables.InducingVariables,
        kernel: gpflow.kernels.Kernel,
        domain: np.ndarray,
        q_mu: np.ndarray,
        q_S: np.ndarray,
        time_dims,
        mark_dims,
        marked,
        *,
        beta0: float = 1e-6,
    ):
        """
        D = number of dimensions: U (#types) x Q (#last events)
        M = size of inducing variables (number of inducing points)

        :param inducing_variable: inducing variables (here only implemented for a gpflow
            .inducing_variables.InducingPoints instance, with Z of shape M x D)
        :param kernel: the kernel (here only implemented for a gpflow.kernels
            .SquaredExponential instance)
        :param domain: lower and upper bounds of (hyper-rectangular) domain
            (D x 2)

        :param q_mu: initial mean vector of the variational distribution q(u)
            (length M)
        :param q_S: how to initialise the covariance matrix of the variational
            distribution q(u)  (M x M)

        :param beta0: a constant offset, corresponding to initial value of the
            prior mean of the GP (but trainable); should be sufficiently large
            so that the GP does not go negative...
        :param num_observations: number of observations of sets of events
            under the distribution
        :param T: end of the observation period

        """
        super().__init__(
            kernel,
            likelihood=None,  # custom likelihood
            num_latent_gps=1,
        )

        self.time_dims = time_dims
        self.Dt = len(time_dims)
        self.mark_dims = mark_dims
        self.marked = marked

        # observation domain  (D x 2)
        # Each dimension d = (u, q) corresponds to qth last event from uth event type
        # d ~ [0, max(time span between q+1 points of same type)]
        self.domain = domain
        if domain.ndim != 2 or domain.shape[1] != 2:
            raise ValueError("domain must be of shape D x 2")

        self.kernel = kernel
        self.inducing_variable = inducing_variable

        self.beta0 = Parameter(beta0, transform=positive(), name="beta0")  # constant mean offset

        # variational approximate Gaussian posterior q(u) = N(u; m, S)
        self.q_mu = Parameter(q_mu, name="q_mu")  # mean vector  (length M)

        # covariance:
        L = np.linalg.cholesky(q_S)  # S = L L^T, with L lower-triangular  (M x M)
        self.q_sqrt = Parameter(L, transform=triangular(), name="q_sqrt")

        self.psi_jitter = 0.0
        self.Kuu_jitter = 1e-6
        self.iter = 0

    @property
    def total_area(self):
        return np.prod(self.domain[:, 1] - self.domain[:, 0])

    def predict_f(self, Xnew, full_cov=False, *, Kuu=None):
        """
        VBPP-specific conditional on the approximate posterior q(u), including a
        constant mean function.
        """
        mean, var = conditional(
            Xnew,
            self.inducing_variable,
            self.kernel,
            self.q_mu[:, None],
            full_cov=full_cov,
            q_sqrt=self.q_sqrt[None, :, :],
        )
        return mean + self.beta0, var

    def elbo(self, abs_evs, rel_at_targets, rel_at_alls, evs_start):
        # Kuu = k(Z, Z)
        Kuu = gpflow.covariances.Kuu(self.inducing_variable, self.kernel, jitter=1e-6)

        # Terms
        data_term = 0
        integral_term = 0
        for rel_at_target in rel_at_targets:
            if len(rel_at_target) > 0:
                data_term += self._elbo_data_term(rel_at_target, Kuu)
        integral_term += self._elbo_integral_term(abs_evs, rel_at_alls, evs_start, Kuu)
        kl_term = self.prior_kl(Kuu)

        # ELBO
        _elbo = data_term - integral_term - kl_term
        # variances, lengthscales = get_hyperparameters(self.kernel)
        # tf.print("variances: ", variances)
        # tf.print("lengthscales: ", lengthscales)
        # if self.marked:
        #     tf.print("mark lengthscales: ", get_mark_lengthscales(self.kernel))
        # tf.print("beta0: ", self.beta0)
        # tf.print("elbo: ", _elbo)
        return _elbo

    def _elbo_data_term(self, rel_ev, Kuu):
        mean_fn, var_fn = self.predict_f(rel_ev, full_cov=False, Kuu=Kuu)
        mean_log_fn_sqr = integrate_log_fn_sqr(mean_fn, var_fn)
        Lfn = tf.reduce_sum(mean_log_fn_sqr)
        return Lfn

    def _var_fx_kxx_term(self, evs_start):
        Dt = len(self.time_dims)
        variances, _ = get_hyperparameters(self.kernel)

        kxx_term = 0
        # Include the last period tN to T
        T = self.domain[0, -1]
        for d in range(Dt):
            kxx_term += (T - evs_start[d]) * variances[d]
        return kxx_term

    def _elbo_integral_term(self, abs_evs, rel_at_alls, evs_start, Kuu):
        # q(f) = GP(f; μ, Σ)
        # Psi, Ψ(z,z') = ∫ K(z,x) K(x,z')
        M = tf.shape(self.inducing_variable.Z)[0]
        D = tf.shape(self.inducing_variable.Z)[1]
        Psi = tf.zeros((M, M), dtype=default_float())
        Phi = tf.zeros(M, dtype=default_float())
        kxx_term = 0
        variances, lengthscales = get_hyperparameters(self.kernel)
        Z_times = tf.gather(self.inducing_variable.Z, self.time_dims, axis=1)
        mark_output = None
        for abs_ev, rel_ev, ev_start in zip(abs_evs, rel_at_alls, evs_start):
            abs_times = tf.gather(abs_ev, [0], axis=1)
            rel_times = tf.gather(rel_ev, self.time_dims, axis=1)
            if self.marked:
                rel_marks = tf.gather(rel_ev, self.mark_dims, axis=1)
                Z_marks = tf.gather(self.inducing_variable.Z, self.mark_dims, axis=1)
                mark_output = get_mark_output(self.kernel, rel_marks, Z_marks)  # (N, M, Dt)
            mask_not_inf_time_float, t1_minus_s, t0_minus_s, s = self._get_Psi_Phi_variables(abs_times, rel_times)
            Psi += tf_calc_Psi_matrix_vec(variances, lengthscales, mask_not_inf_time_float, t1_minus_s, t0_minus_s, s,
                                          Z_times, mark_output)
            Phi += tf_calc_Phi_SqExp_vec(variances, lengthscales, mask_not_inf_time_float, t1_minus_s, t0_minus_s,
                                         Z_times, mark_output)
            kxx_term += self._var_fx_kxx_term(ev_start)

        # int_expect_fx_sqr = m^T Kzz⁻¹ Ψ Kzz⁻¹ m
        # = (Kzz⁻¹ m)^T Ψ (Kzz⁻¹ m)
        # Kzz = R R^T
        R = tf.linalg.cholesky(Kuu)

        # Kzz⁻¹ m = R^-T R⁻¹ m
        # Linv_m = R⁻¹ m
        Rinv_m = tf.linalg.triangular_solve(R, self.q_mu[:, None], lower=True)

        # R⁻¹ Ψ R^-T
        # = (R⁻¹ Ψ) R^-T
        Rinv_Psi = tf.linalg.triangular_solve(R, Psi, lower=True)
        # # = (Rinv_Ψ) R^-T = (R⁻¹ Rinv_Ψ^T)^T
        Rinv_Psi_RinvT = tf.linalg.triangular_solve(R, tf.transpose(Rinv_Psi), lower=True)

        int_mean_f_sqr = tf_vec_mat_vec_mul(Rinv_m, Rinv_Psi_RinvT, Rinv_m)

        Rinv_L = tf.linalg.triangular_solve(R, self.q_sqrt, lower=True)
        Rinv_L_LT_RinvT = tf.matmul(Rinv_L, Rinv_L, transpose_b=True)

        # int_var_fx = ∫ x_d(t) γ_d  + trace_terms
        # trace_terms = - Tr(Kzz⁻¹ Ψ) + Tr(Kzz⁻¹ S Kzz⁻¹ Ψ)
        trace_terms = tf.reduce_sum(
            (Rinv_L_LT_RinvT - tf.eye(self.inducing_variable.num_inducing, dtype=default_float()))
            * Rinv_Psi_RinvT
        )

        int_var_f = kxx_term + trace_terms

        f_term = int_mean_f_sqr + int_var_f

        # λ = E_f{(f + β₀)**2}
        #   = (E_f)^2 + var_f + 2 f β₀ + β₀^2
        #   = f_term + int_cross_terms + betas_term
        Kuu_inv_m = tf.linalg.triangular_solve(tf.transpose(R), Rinv_m, lower=False)

        int_cross_term = 2 * self.beta0 * tf_vec_dot(Phi, Kuu_inv_m)

        beta_term = tf.square(self.beta0) * self.total_area

        int_lambda = f_term + int_cross_term + beta_term

        return int_lambda

    def prior_kl(self, Kuu):
        """
        KL(q || p), where q(u) = N(m, S) and p(u) = N(0, Kuu)
        """
        return kullback_leiblers.gauss_kl(self.q_mu[:, None], self.q_sqrt[None, :, :], K=Kuu)

    @staticmethod
    def _get_Psi_Phi_variables(abs_times, rel_times):
        mask_not_inf_time = tf.logical_not(tf.math.is_inf(rel_times))  # (N, D) == I
        mask_not_inf_time_float = tf.cast(mask_not_inf_time, tf.float64)  # (N, D)
        # mask_float: (N, 1, 1, D, D)
        # Time mask also contains the masking information of mark mask, as invalid times/marks both set to inf.
        rel_times_masked = tf.where(mask_not_inf_time, rel_times, to_default_float(0.0))
        t1_minus_s = rel_times_masked  # (N, Dt)
        t0_minus_s = tf.where(mask_not_inf_time,
                              rel_times_masked - tf.math.reduce_min(rel_times, axis=1, keepdims=True),
                              to_default_float(0.0))  # (N, Dt)
        s = tf.where(mask_not_inf_time, abs_times - t0_minus_s, to_default_float(0.0))  # (N, Dt)
        return mask_not_inf_time_float, t1_minus_s, t0_minus_s, s

    def predict_lambda_and_percentiles(self, Xnew, lower=5, upper=95):
        """
        Computes mean value of intensity and lower and upper percentiles.
        `lower` and `upper` must be between 0 and 100.
        """
        # f ~ Normal(mean_f, var_f)
        mean_f, var_f = self.predict_f(Xnew)
        # λ = E[f²] = E[f]² + Var[f]
        lambda_mean = mean_f ** 2 + var_f
        # g = f/√var_f ~ Normal(mean_f/√var_f, 1)
        # g² = f²/var_f ~ χ²(k=1, λ=mean_f²/var_f) non-central chi-squared
        m2ov = mean_f ** 2 / var_f
        if tf.reduce_any(m2ov > 10e3):
            raise ValueError("scipy.stats.ncx2.ppf() flatlines for nc > 10e3")
        f2ov_lower = ncx2.ppf(lower / 100, df=1, nc=m2ov)
        f2ov_upper = ncx2.ppf(upper / 100, df=1, nc=m2ov)
        # f² = g² * var_f
        lambda_lower = f2ov_lower * var_f
        lambda_upper = f2ov_upper * var_f
        return lambda_mean, lambda_lower, lambda_upper

    def maximum_log_likelihood_objective(self, abs_evs, rel_evs) -> tf.Tensor:
        return self.elbo(abs_evs, rel_evs)

    def predict_lambda(self, Xnew):
        """
        Returns expected value of the intensity function lambda(.): E[lambda]
        """
        mean, var = self.predict_f(Xnew)
        return mean - self.beta0, tf.square(mean) + var

    def predict_integral_term(self, abs_ev, rel_at_all, ev_start):
        K = gpflow.covariances.Kuu(self.inducing_variable, self.kernel, jitter=1e-6)
        integral_term = self._elbo_integral_term(abs_ev, rel_at_all, ev_start, K)
        return integral_term
