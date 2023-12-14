import gpflow.mean_functions
import numpy as np
import gpflow as gpf
# noinspection PyPackageRequirements
import tensorflow as tf
from gpflow import logdensities
from gpflow.conditionals.util import sample_mvn
from gpflow.logdensities import multivariate_normal
from gpflow.models.util import data_input_to_tensor

from models.outcome.piecewise_se.kernel import HierarchicalMarkCoregion


class TRSharedMarkedHier(gpf.models.GPR):

    def __init__(self, data, t, T, baseline_kernels, treatment_base_kernel, mean_functions, patient_ids, use_bias,
                 noise_variance=1.0,
                 separating_interval=240.0,
                 train_noise=True):
        """

        :param data: (X,Y): both list of (N,1) arrays
        :param t: list of (M, 2)
        :param T:
        :param baseline_kernels: kernel list
        :param treatment_base_kernel: single kernel
        :param mean_functions: mean functions list
        :param patient_ids: List[int]
        :param noise_variance:
        :param separating_interval:
        """
        X, Y = data
        self.N = len(X)
        super().__init__((X[0], Y[0]), None, None, noise_variance)
        self.data = data_input_to_tensor(data)
        self.mean_functions = mean_functions
        self.baseline_kernels = baseline_kernels
        self.T = T
        self.separating_interval = separating_interval
        # TODO: switched likelihood
        self.likelihood = [gpflow.likelihoods.Gaussian(noise_variance) for _ in range(self.N)]
        for lik in self.likelihood:
            gpflow.utilities.set_trainable(lik, train_noise)
        self.intervals = tf.cast(tf.stack([n * self.separating_interval for n in range(self.N)]), tf.float64)
        self.X_separated, self.t_separated = self.prepare_treatment_input(X, t)
        self.ft_data = self.prepare_ft_vec(self.X_separated, self.t_separated, self.T)
        self.patient_ids = tf.convert_to_tensor(patient_ids, dtype=tf.int32)
        t_lengths = [tf.shape(ti)[0] for ti in t]
        self.treatment_patient_idx = tf.repeat(tf.range(self.N), t_lengths)
        self.treatment_kc = HierarchicalMarkCoregion(output_dim=tf.shape(self.t_separated)[0], rank=1,
                                                     num_patients=self.N,
                                                     treatment_patient_idx=self.treatment_patient_idx,
                                                     beta0=0.1, beta1=0.1, sigma_raw=0.1,
                                                     active_dims=[1, 2], use_bias=use_bias)
        self.treatment_kc.kappa.assign(np.ones(self.treatment_kc.kappa.shape) * 1e-12)
        gpf.set_trainable(self.treatment_kc.kappa, False)
        self.treatment_base_kernel = treatment_base_kernel
        self.treatment_kernel = self.treatment_base_kernel * self.treatment_kc
        self.block_diag_scatter_ids_train = self.get_block_diag_scatter_ids(X)

    def prepare_treatment_input(self, Xnew, tnew):
        N = len(Xnew)
        xnew_lengths = [tf.shape(xi)[0] for xi in Xnew]
        Xnew_flat = tf.concat(Xnew, axis=0)
        intervals = tf.cast(tf.stack([n * self.separating_interval for n in range(N)]), tf.float64)
        Xnew_separated = Xnew_flat + tf.reshape(tf.repeat(intervals, xnew_lengths), (-1, 1))
        if tnew is not None:
            tnew_lengths = [tf.shape(ti)[0] for ti in tnew]
            tnew_patient_idx = tf.repeat(tf.range(len(tnew)), tnew_lengths)
            tnew_flat = tf.concat([ti[:, 0] for ti in tnew], axis=0)
            mark_flat = tf.concat([ti[:, 1] for ti in tnew], axis=0)
            tnew_flat = tnew_flat + tf.repeat(intervals, tnew_lengths)
            tnew_separated = tf.stack([tnew_flat, mark_flat], axis=1)
        else:
            tnew_separated = []
        return Xnew_separated, tnew_separated

    @staticmethod
    def prepare_ft_vec(X, t, T, Xnew=None, tnew=None):
        """

        :param X: (N, 1)
        :param t: (Nt, 2)
        :param T: float
        :param Xnew: (N2, 1)
        :param tnew: (Nt2, 2)
        :return:
        """
        if Xnew is None:
            Xnew = X
        if tnew is None:
            tnew = t
        t_row = tf.reshape(t[:, 0], (1, -1))
        deltaX1 = X - t_row  # dim: (N1, Nt)
        mask1 = tf.math.logical_and(deltaX1 > 0.0, deltaX1 <= T)   # 2d mask
        y_coord = tf.experimental.numpy.nonzero(tf.transpose(mask1))[1]
        #
        tnew_row = tf.reshape(tnew[:, 0], (1, -1))
        deltaX2 = Xnew - tnew_row  # dim: (N1, Nt)
        mark2 = tf.gather(tnew, 1, axis=1)
        mask2 = tf.math.logical_and(deltaX2 > 0.0, deltaX2 <= T)
        x_coord = tf.experimental.numpy.nonzero(tf.transpose(mask2))[1]
        # Here, x and y follows Cartesian axis notation
        # yy: First axis in matrix notation
        # xx: Second axis in matrix notation
        xx, yy = tf.meshgrid(x_coord, y_coord)
        yy_flat = tf.reshape(yy, (-1,))
        xx_flat = tf.reshape(xx, (-1,))
        scatter_ids_2d = tf.transpose(tf.stack([yy_flat, xx_flat]))
        scatter_ids_1d = tf.reshape(x_coord, (-1, 1))
        # We need deltaX2s for kmn and knn.
        # We need deltaX1s for kmm. But, this is handled before training for now.
        # Perhaps a better solution is to return them all each time.
        n_deltaX2 = tf.reduce_sum(tf.cast(mask2, dtype=tf.int64), axis=0)  # Number of obs in each treat region
        # Extract X where treatment affects to reduce computation size
        deltaXs = tf.transpose(deltaX2)[tf.transpose(mask2)]  # Mask operation by column first
        # small perturbation for uniqueness
        mark2 += tf.random.normal(tf.shape(mark2), mean=0.0, stddev=0.001, dtype=tf.float64)
        deltaX_mark = tf.repeat(mark2, n_deltaX2)
        deltaX_treat_idx = tf.repeat(tf.range(tf.shape(n_deltaX2)[0], dtype=tf.float64), n_deltaX2)
        deltaXs = tf.stack([deltaXs, deltaX_treat_idx, deltaX_mark], axis=1)  # [X, X_type]
        #
        return deltaXs, scatter_ids_2d, scatter_ids_1d

    @staticmethod
    def get_block_diag_scatter_ids(X, X2=None):
        if X2 is None:
            X2 = X
        offset1 = 0
        offset2 = 0
        indices = []
        N = len(X)
        for n in range(N):
            Ni1 = tf.shape(X[n])[0]
            Ni2 = tf.shape(X2[n])[0]
            xx, yy = tf.meshgrid(tf.range(Ni2), tf.range(Ni1))
            yy_flat = tf.reshape(yy, (-1,))
            xx_flat = tf.reshape(xx, (-1,))
            indices_i = tf.transpose(tf.stack([yy_flat + offset1, xx_flat + offset2]))
            offset1 += Ni1
            offset2 += Ni2
            indices.append(indices_i)

        indices = tf.concat(indices, axis=0)
        return indices

    def add_switched_noise_cov(self, K: tf.Tensor) -> tf.Tensor:
        """
        Returns K + σ² I, where σ² is the likelihood noise variance (scalar),
        and I is the corresponding identity matrix.
        """
        X, _ = self.data
        x_lengths = [tf.shape(xi)[0] for xi in X]
        noise_variances = tf.stack([lik.variance for lik in self.likelihood])
        s_diag = tf.repeat(noise_variances, x_lengths)
        k_diag = tf.linalg.diag_part(K)
        return tf.linalg.set_diag(K, k_diag + s_diag)

    def log_marginal_likelihood(self) -> tf.Tensor:
        r"""
        Computes the log marginal likelihood.

        .. math::
            \log p(Y | \theta).

        """
        X, Y = self.data
        xs_train, scatter_ids_train, _ = self.ft_data
        K = self.K_sum(X, xs_train, self.t_separated,
                       baseline_scatter_ids=self.block_diag_scatter_ids_train,
                       treatment_scatter_ids=scatter_ids_train)
        ks = self.add_switched_noise_cov(K)
        ks += 1e-9 * tf.eye(ks.shape[0], dtype=ks.dtype)
        L = tf.linalg.cholesky(ks)
        m = tf.concat([mf(Xi) for mf, Xi in zip(self.mean_functions, X)], axis=0)

        # [R,] log-likelihoods for each independent dimension of Y
        Y_all = tf.cast(tf.concat(Y, axis=0), tf.float64)
        log_prob = multivariate_normal(Y_all, m, L)
        return tf.reduce_sum(log_prob)

    def Kb(self, X, scatter_ids, X2=None):
        if X2 is None:
            X2 = X
        sparse_flattened = tf.concat([tf.reshape(kernel_bi(x1i, x2i), (-1,))
                                      for kernel_bi, x1i, x2i in zip(self.baseline_kernels, X, X2)], axis=0)

        N1_total = sum([tf.shape(xi)[0] for xi in X])
        N2_total = sum([tf.shape(xi)[0] for xi in X2])
        sparse_cov = tf.sparse.SparseTensor(tf.cast(scatter_ids, tf.int64), sparse_flattened, [N1_total, N2_total])
        Kb = tf.sparse.to_dense(sparse_cov)
        return Kb

    def Kb_diag(self, X):
        return tf.concat([kernel_bi(X2i, full_cov=False) for kernel_bi, X2i in zip(self.baseline_kernels, X)], axis=0)

    def K_sum(self, X, xs, ts, baseline_scatter_ids, treatment_scatter_ids, X2=None, xs2=None):
        if X2 is None:
            X2 = X
        if xs2 is None:
            xs2 = xs
        Ksum = self.Kb(X, scatter_ids=baseline_scatter_ids, X2=X2)
        if len(ts) > 0:
            Kt = self.treatment_kernel(xs, xs2)
            Ksum = tf.tensor_scatter_nd_add(Ksum, treatment_scatter_ids, tf.reshape(Kt, [-1]))
        return Ksum

    def K_sum_diag(self, X2, xs2, scatter_ids_1d):
        Kb = self.Kb_diag(X2)
        Kt = self.treatment_kernel(xs2, full_cov=False)
        K_sum_diag = tf.tensor_scatter_nd_add(Kb, scatter_ids_1d, Kt)
        return K_sum_diag

    def Kt(self, X, xs, scatter_ids, X2=None, xs2=None):
        if X2 is None:
            X2 = X
        if xs2 is None:
            xs2 = xs
        Kt = self.treatment_kernel(xs, xs2)
        # Work around for tf.function compile, it complains about dynamic shape
        K_scatter = tf.matmul(tf.zeros_like(X), tf.zeros_like(X2), transpose_b=True)
        K_scatter = tf.tensor_scatter_nd_add(K_scatter, scatter_ids, tf.reshape(Kt, [-1]))
        return K_scatter

    def Kt_diag(self, X, xs, scatter_ids_1d):
        Kt = self.treatment_kernel(xs, full_cov=False)
        # Work around for tf.function compile, it complains about dynamic shape
        K_scatter_diag = tf.reshape(tf.zeros_like(X), (-1,))
        K_sum_diag = tf.tensor_scatter_nd_add(K_scatter_diag, scatter_ids_1d, Kt)
        return K_sum_diag

    def predict_ft_w_tnew(
            self, Xnew, tnew, tnew_porder, full_cov: bool = True, full_output_cov: bool = False
    ):
        r"""
        """
        Xnew_separated, tnew_separated = self.prepare_treatment_input(Xnew, tnew)
        if len(tnew_separated) == 0 or len(self.t_separated) == 0:
            return tf.zeros_like(Xnew_separated), tf.zeros_like(Xnew_separated)

        X, Y = self.data
        xs_train, scatter_ids_mm, _ = self.ft_data
        self.treatment_kc.set_t_pidx(self.treatment_patient_idx)
        self.treatment_kc.set_t_pidx2(self.treatment_patient_idx)
        kmm = self.K_sum(X, xs_train, self.t_separated,
                         baseline_scatter_ids=self.block_diag_scatter_ids_train,
                         treatment_scatter_ids=scatter_ids_mm)
        kmm_plus_s = self.add_switched_noise_cov(kmm)

        # Needed for Kmn, Knn
        xsnew, scatter_ids_mn, scatter_ids_nn_diag = self.prepare_ft_vec(self.X_separated, self.t_separated, self.T,
                                                                         Xnew=Xnew_separated, tnew=tnew_separated)
        self.treatment_kc.set_t_pidx(self.treatment_patient_idx)
        self.treatment_kc.set_t_pidx2(tnew_porder)
        kmn = self.Kt(self.X_separated, xs_train, scatter_ids_mn, Xnew_separated, xsnew)
        self.treatment_kc.set_t_pidx(tnew_porder)
        self.treatment_kc.set_t_pidx2(tnew_porder)
        if full_cov:
            _, scatter_ids_nn, _ = self.prepare_ft_vec(Xnew_separated, tnew_separated, self.T,
                                                       Xnew=Xnew_separated, tnew=tnew_separated)
            knn = self.Kt(Xnew_separated, xsnew, scatter_ids_nn)
        else:
            knn = self.Kt_diag(Xnew_separated, xsnew, scatter_ids_nn_diag)

        m = tf.concat([mf(Xi) for mf, Xi in zip(self.mean_functions, X)], axis=0)
        Y_all = tf.cast(tf.concat(Y, axis=0), tf.float64)
        err = Y_all - m
        conditional = gpf.conditionals.base_conditional
        f_mean_zero, f_var = conditional(
            kmn, kmm_plus_s, knn, err, full_cov=full_cov, white=False
        )  # [N, P], [N, P] or [P, N, N]
        mnew = tf.concat([mf(Xi) for mf, Xi in zip(self.mean_functions, Xnew)], axis=0)
        return f_mean_zero+mnew, f_var

    def predict_baseline(
            self, Xnew, full_cov: bool = False, full_output_cov: bool = False
    ):
        r"""
        """
        X, Y = self.data
        xs_train, scatter_ids_mm, _ = self.ft_data
        self.treatment_kc.set_t_pidx(self.treatment_patient_idx)
        self.treatment_kc.set_t_pidx2(self.treatment_patient_idx)
        kmm = self.K_sum(X, xs_train, self.t_separated,
                         baseline_scatter_ids=self.block_diag_scatter_ids_train,
                         treatment_scatter_ids=scatter_ids_mm)
        kmm_plus_s = self.add_switched_noise_cov(kmm)

        # Needed for Kmn, Knn
        baseline_scatter_ids_mn = self.get_block_diag_scatter_ids(X, Xnew)
        kmn = self.Kb(X, scatter_ids=baseline_scatter_ids_mn, X2=Xnew)
        if full_cov:
            baseline_scatter_ids_nn = self.get_block_diag_scatter_ids(Xnew, Xnew)
            knn = self.Kb(Xnew, baseline_scatter_ids_nn)
        else:
            knn = self.Kb_diag(Xnew)

        m = tf.concat([mf(Xi) for mf, Xi in zip(self.mean_functions, X)], axis=0)
        Y_all = tf.cast(tf.concat(Y, axis=0), tf.float64)
        err = Y_all - m
        conditional = gpf.conditionals.base_conditional
        f_mean_zero, f_var = conditional(
            kmn, kmm_plus_s, knn, err, full_cov=full_cov, white=False
        )  # [N, P], [N, P] or [P, N, N]
        mnew = tf.concat([mf(Xi) for mf, Xi in zip(self.mean_functions, Xnew)], axis=0)
        f_mean = f_mean_zero + mnew
        return f_mean, f_var

    def predict_ft_w_tnew_conditional(
            self, X, Y, t, t_porder, Xnew, tnew, tnew_porder, full_cov: bool = False, full_output_cov: bool = False
    ):
        r"""
        """
        X_separated, t_separated = self.prepare_treatment_input(X, t)
        xs_train, scatter_ids_mm, _ = self.prepare_ft_vec(X_separated, t_separated, self.T)
        scatter_ids_mm_diag = self.get_block_diag_scatter_ids(X)
        self.treatment_kc.set_t_pidx(t_porder)
        self.treatment_kc.set_t_pidx2(t_porder)
        #
        Xnew_separated, tnew_separated = self.prepare_treatment_input(Xnew, tnew)
        if len(tnew_separated) == 0 or len(self.t_separated) == 0:
            return tf.zeros_like(Xnew_separated), tf.zeros_like(Xnew_separated)

        self.treatment_kc.set_t_pidx(t_porder)
        self.treatment_kc.set_t_pidx2(t_porder)
        kmm = self.K_sum(X, xs_train, t_separated,
                         baseline_scatter_ids=scatter_ids_mm_diag,
                         treatment_scatter_ids=scatter_ids_mm)
        kmm_plus_s = self.add_switched_noise_cov(kmm)

        # Needed for Kmn, Knn
        xsnew, scatter_ids_mn, scatter_ids_nn_diag = self.prepare_ft_vec(X_separated, t_separated, self.T,
                                                                         Xnew=Xnew_separated, tnew=tnew_separated)
        self.treatment_kc.set_t_pidx(t_porder)
        self.treatment_kc.set_t_pidx2(tnew_porder)
        kmn = self.Kt(self.X_separated, xs_train, scatter_ids_mn, Xnew_separated, xsnew)
        self.treatment_kc.set_t_pidx(tnew_porder)
        self.treatment_kc.set_t_pidx2(tnew_porder)
        if full_cov:
            _, scatter_ids_nn, _ = self.prepare_ft_vec(Xnew_separated, tnew_separated, self.T,
                                                       Xnew=Xnew_separated, tnew=tnew_separated)
            knn = self.Kt(Xnew_separated, xsnew, scatter_ids_nn)
        else:
            knn = self.Kt_diag(Xnew_separated, xsnew, scatter_ids_nn_diag)

        m = tf.concat([mf(Xi) for mf, Xi in zip(self.mean_functions, X)], axis=0)
        Y_all = tf.cast(tf.concat(Y, axis=0), tf.float64)
        err = Y_all - m
        conditional = gpf.conditionals.base_conditional
        f_mean_zero, f_var = conditional(
            kmn, kmm_plus_s, knn, err, full_cov=full_cov, white=False
        )  # [N, P], [N, P] or [P, N, N]
        mnew = tf.concat([mf(Xi) for mf, Xi in zip(self.mean_functions, Xnew)], axis=0)
        return f_mean_zero+mnew, f_var

    def predict_baseline_conditional(
            self, X, Y, t, t_porder, Xnew, full_cov: bool = False, full_output_cov: bool = False
    ):
        r"""
        """
        X_separated, t_separated = self.prepare_treatment_input(X, t)
        xs_train, scatter_ids_mm, _ = self.prepare_ft_vec(X_separated, t_separated, self.T)
        block_diag_scatter_ids_train = self.get_block_diag_scatter_ids(X)
        self.treatment_kc.set_t_pidx(t_porder)
        self.treatment_kc.set_t_pidx2(t_porder)
        kmm = self.K_sum(X, xs_train, t_separated,
                         baseline_scatter_ids=block_diag_scatter_ids_train,
                         treatment_scatter_ids=scatter_ids_mm)
        kmm_plus_s = self.add_switched_noise_cov(kmm)

        # Needed for Kmn, Knn
        baseline_scatter_ids_mn = self.get_block_diag_scatter_ids(X, Xnew)
        kmn = self.Kb(X, scatter_ids=baseline_scatter_ids_mn, X2=Xnew)
        if full_cov:
            baseline_scatter_ids_nn = self.get_block_diag_scatter_ids(Xnew, Xnew)
            knn = self.Kb(Xnew, baseline_scatter_ids_nn)
        else:
            knn = self.Kb_diag(Xnew)

        m = tf.concat([mf(Xi) for mf, Xi in zip(self.mean_functions, X)], axis=0)
        Y_all = tf.cast(tf.concat(Y, axis=0), tf.float64)
        err = Y_all - m
        conditional = gpf.conditionals.base_conditional
        f_mean_zero, f_var = conditional(
            kmn, kmm_plus_s, knn, err, full_cov=full_cov, white=False
        )  # [N, P], [N, P] or [P, N, N]
        mnew = tf.concat([mf(Xi) for mf, Xi in zip(self.mean_functions, Xnew)], axis=0)
        f_mean = f_mean_zero + mnew
        return f_mean, f_var

    def predict_log_density_for_patients(
        self, Xnew, Ynew, tnew, tnew_porder
    ) -> tf.Tensor:
        """
        Compute the log density of the data at the new data points.
        """
        f_mean, f_var = self.predict_ft_w_tnew(Xnew, tnew, tnew_porder)
        x_lengths = [tf.shape(xi)[0] for xi in Xnew]
        variance_patient = tf.stack([lik.variance for lik in self.likelihood])
        variance_vec = tf.reshape(tf.repeat(variance_patient, x_lengths), (-1, 1))
        Ynew_concat = tf.cast(tf.concat(Ynew, axis=0), tf.float64)
        ll = tf.reduce_sum(logdensities.gaussian(Ynew_concat, f_mean, f_var + variance_vec), axis=-1)
        return ll

    def predict_ft_single(
            self, Xnew, tnew, tnew_pidx, full_cov: bool = False, full_output_cov: bool = False,
    ):
        r"""
        Predict single ft.
        """
        if len(tnew) == 0 or len(self.t_separated) == 0:
            return tf.zeros_like(Xnew), tf.zeros_like(Xnew)

        X, Y = self.data
        self.treatment_kc.set_t_pidx(self.treatment_patient_idx)
        self.treatment_kc.set_t_pidx2(self.treatment_patient_idx)
        xs_train, scatter_ids_mm, _ = self.ft_data
        kmm = self.K_sum(X, xs_train, self.t_separated,
                         baseline_scatter_ids=self.block_diag_scatter_ids_train,
                         treatment_scatter_ids=scatter_ids_mm)
        kmm_plus_s = self.add_switched_noise_cov(kmm)

        # Needed for Kmn, Knn
        xsnew, scatter_ids_mn, scatter_ids_nn_diag = self.prepare_ft_vec(self.X_separated, self.t_separated, self.T,
                                                                         Xnew=Xnew, tnew=tnew)
        self.treatment_kc.set_t_pidx(self.treatment_patient_idx)
        self.treatment_kc.set_t_pidx2(tnew_pidx)
        kmn = self.Kt(self.X_separated, xs_train, scatter_ids_mn, Xnew, xsnew)
        self.treatment_kc.set_t_pidx(tnew_pidx)
        self.treatment_kc.set_t_pidx2(tnew_pidx)
        if full_cov:
            _, scatter_ids_nn, _ = self.prepare_ft_vec(Xnew, tnew, self.T, Xnew=Xnew, tnew=tnew)
            knn = self.Kt(Xnew, xsnew, scatter_ids_nn)
        else:
            knn = self.Kt_diag(Xnew, xsnew, scatter_ids_nn_diag)

        m = tf.concat([mf(Xi) for mf, Xi in zip(self.mean_functions, X)], axis=0)
        Y_all = tf.cast(tf.concat(Y, axis=0), tf.float64)
        err = Y_all - m
        conditional = gpf.conditionals.base_conditional
        f_mean_zero, f_var = conditional(
            kmn, kmm_plus_s, knn, err, full_cov=full_cov, white=False
        )  # [N, P], [N, P] or [P, N, N]
        return f_mean_zero, f_var

    def predict_ft_single_for_patient(
            self, Xnew, tnew, pidx_rd, full_cov: bool = True, full_output_cov: bool = False,
    ):
        r"""
        Predict single ft.

        pidx: Patient idx corresponding to real data: pidx \in [0, ..., 13]
        """
        if len(tnew) == 0 or len(self.t_separated) == 0:
            return tf.zeros_like(Xnew), tf.zeros_like(Xnew)
        porder = tf.argmax(tf.cast(tf.equal(self.patient_ids, pidx_rd), tf.int32))
        tnew_pidx = np.ones(tnew.shape[0], dtype=np.int32) * porder
        return self.predict_ft_single(Xnew, tnew, tnew_pidx, full_cov, full_output_cov)

    def predict_log_density_for_patient(
        self, Xnew, Ynew, tnew, pidx_rd
    ) -> tf.Tensor:
        """
        Compute the log density of the data at the new data points.
        """
        f_mean, f_var = self.predict_ft_single_for_patient(Xnew, tnew, pidx_rd, full_cov=False, full_output_cov=False)
        porder = tf.argmax(tf.cast(tf.equal(self.patient_ids, pidx_rd), tf.int32))
        lls = tf.stack([lik.predict_log_density(f_mean, f_var, Ynew) for lik in self.likelihood])
        return tf.gather(lls, porder)

    def predict_baseline_samples(
        self,
        Xnew,
        num_samples: int,
        full_cov: bool = True,
    ) -> tf.Tensor:
        """
        Produce samples from the posterior latent function(s) at the input points.

        :param Xnew: InputData
            Input locations at which to draw samples, shape [..., N, D]
            where N is the number of rows and D is the input dimension of each point.
        :param num_samples:
            Number of samples to draw.
            If `None`, a single sample is drawn and the return shape is [..., N, P],
            for any positive integer the return shape contains an extra batch
            dimension, [..., S, N, P], with S = num_samples and P is the number of outputs.
        :param full_cov:
            If True, draw correlated samples over the inputs. Computes the Cholesky over the
            dense covariance matrix of size [num_data, num_data].
            If False, draw samples that are uncorrelated over the inputs.

        Currently, the method does not support `full_output_cov=True` and `full_cov=True`.
        """
        mean, cov = self.predict_baseline(Xnew, full_cov=full_cov, full_output_cov=False)
        # mean: [..., N, P]
        # cov: [..., P, N, N]
        mean_for_sample = tf.linalg.adjoint(mean)  # [..., P, N]
        samples = sample_mvn(
            mean_for_sample, cov, full_cov, num_samples=num_samples
        )  # [..., (S), P, N]
        samples = tf.linalg.adjoint(samples)  # [..., (S), N, P]
        return samples  # [..., (S), N, P]

    def predict_ft_samples(
        self,
        Xnew,
        tnew,
        num_samples,
        full_cov: bool = True,
    ) -> tf.Tensor:
        """
        Produce samples from the posterior latent function(s) at the input points.

        :param Xnew: InputData
            Input locations at which to draw samples, shape [..., N, D]
            where N is the number of rows and D is the input dimension of each point.
        :param tnew:
        :param num_samples:
            Number of samples to draw.
            If `None`, a single sample is drawn and the return shape is [..., N, P],
            for any positive integer the return shape contains an extra batch
            dimension, [..., S, N, P], with S = num_samples and P is the number of outputs.
        :param full_cov:
            If True, draw correlated samples over the inputs. Computes the Cholesky over the
            dense covariance matrix of size [num_data, num_data].
            If False, draw samples that are uncorrelated over the inputs.

        Currently, the method does not support `full_output_cov=True` and `full_cov=True`.
        """
        mean, cov = self.predict_ft(Xnew, tnew, full_cov=full_cov, full_output_cov=False)
        # mean: [..., N, P]
        # cov: [..., P, N, N]
        mean_for_sample = tf.linalg.adjoint(mean)  # [..., P, N]
        samples = sample_mvn(
            mean_for_sample, cov, full_cov, num_samples=num_samples
        )  # [..., (S), P, N]
        samples = tf.linalg.adjoint(samples)  # [..., (S), N, P]
        return samples  # [..., (S), N, P]
