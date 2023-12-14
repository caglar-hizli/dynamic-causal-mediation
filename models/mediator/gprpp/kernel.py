import gpflow
import numpy as np
import tensorflow as tf
from gpflow import default_float
from gpflow.kernels import SquaredExponential
from gpflow.utilities import to_default_float, positive
from gpflow.utilities.ops import square_distance

EPS = 1e-12  #  if there are rows all np.inf


class MaskedSE(gpflow.kernels.Kernel):
    def __init__(self, variance, lengthscales, **kwargs):
        super().__init__(**kwargs)
        self.variance = gpflow.Parameter(variance, transform=positive())
        self.lengthscales = gpflow.Parameter(lengthscales, transform=positive())
        self._validate_ard_active_dims(self.lengthscales)

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        mask_X = tf.math.is_inf(X)
        X_masked = tf.where(mask_X, to_default_float(0.0), X)
        mask_X2 = tf.math.is_inf(X2)
        X2_masked = tf.where(mask_X2, to_default_float(0.0), X2)
        X_scaled = X_masked / self.lengthscales
        X2_scaled = X2_masked / self.lengthscales
        r2 = square_distance(X_scaled, X2_scaled)
        K_ = self.variance * tf.exp(-0.5 * r2)
        K_ = tf.where(mask_X, to_default_float(0.0), K_)
        K_ = tf.where(tf.transpose(tf.math.is_inf(X2)), to_default_float(0.0), K_)
        return K_  # this returns a 2D tensor

    def K_diag(self, X):
        mask_X_inf = tf.reshape(tf.math.is_inf(X), (-1,))
        return tf.where(mask_X_inf, to_default_float(EPS), tf.squeeze(self.variance))  # this returns a 1D tensor


class MaskedMarkedSE(gpflow.kernels.Stationary):
    def __init__(self, variance, lengthscales, **kwargs):
        super().__init__(**kwargs)
        if len(lengthscales) != 2:
            raise ValueError('The lengthscale must be 2-dim for marked PPs!')

        self.variance = gpflow.Parameter(variance, transform=positive())
        self.lengthscales = gpflow.Parameter(lengthscales, transform=positive())
        self._validate_ard_active_dims(self.lengthscales)

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        mask_X = tf.math.reduce_any(tf.math.is_inf(X), axis=1, keepdims=True)  # (N1, 1)
        X_masked = tf.where(mask_X, to_default_float(0.0), X)  # (N1, 2)
        mask_X2 = tf.math.reduce_any(tf.math.is_inf(X2), axis=1, keepdims=True)  # (N2, 1)
        X2_masked = tf.where(mask_X2, to_default_float(0.0), X2)  # (N2, 2)
        X_scaled = X_masked / self.lengthscales
        X2_scaled = X2_masked / self.lengthscales
        r2 = square_distance(X_scaled, X2_scaled)
        K_ = self.variance * tf.exp(-0.5 * r2)  # (N1, N2)
        K_ = tf.where(mask_X, to_default_float(0.0), K_)
        K_ = tf.where(tf.transpose(mask_X2), to_default_float(0.0), K_)
        return K_  # this returns a 2D tensor

    def K_diag(self, X):
        mask_X_inf = tf.math.reduce_any(tf.math.is_inf(X), axis=1)
        return tf.where(mask_X_inf, to_default_float(EPS), tf.squeeze(self.variance))  # this returns a 1D tensor

    def get_mark_lengthscale(self):
        return self.lengthscales[1]

    def get_time_lengthscale(self):
        return self.lengthscales[0]

    def get_mark_covariance(self, M, M2):
        """
        Kernel output only form the mark dimension (dim=[1])

        :param M: (N1, 1) array of marks. All values expected to be finite.
        :param M2: (N2, 2) array of marks. All values expected to be finite.
        """
        X_scaled = M / self.get_mark_lengthscale()
        X2_scaled = M2 / self.get_mark_lengthscale()
        r2 = square_distance(X_scaled, X2_scaled)
        return tf.exp(-0.5 * r2)  # (N1, N2)


def get_hyperparameters(kernel):
    float_type = default_float()
    if isinstance(kernel, MaskedMarkedSE):
        variances = tf.reshape(tf.cast(kernel.variance, float_type), [1])  # dim: (1,)
        lengthscales = tf.reshape(tf.cast(kernel.get_time_lengthscale(), float_type), [1])  # dim: (1,)
    elif isinstance(kernel, MaskedSE) or isinstance(kernel, SquaredExponential):
        variances = tf.reshape(tf.cast(kernel.variance, float_type), [1])  # dim: (1,)
        lengthscales = tf.reshape(tf.cast(kernel.lengthscales, float_type), [1])  # dim: (1,)
    elif isinstance(kernel, gpflow.kernels.base.ReducingCombination):
        variances = []
        lengthscales = []
        for k in kernel.kernels:
            variances.append(tf.cast(k.variance, dtype=float_type))
            not_marked = isinstance(k, MaskedSE) or isinstance(k, SquaredExponential)
            ell = k.lengthscales if not_marked else k.get_time_lengthscale()
            lengthscales.append(tf.cast(ell, dtype=float_type))
        variances = tf.stack(variances)  # dim: (Q_a+Q_o,)
        lengthscales = tf.stack(lengthscales)  # dim: (Q_a+Q_o,)
    else:
        raise NotImplementedError('Not implemented for the kernel configuration!')
    return variances, lengthscales


def get_mark_lengthscales(kernel):
    float_type = default_float()
    if isinstance(kernel, MaskedMarkedSE):
        lengthscales = tf.reshape(tf.cast(kernel.get_mark_lengthscale(), float_type), [1])  # dim: (1,)
    elif isinstance(kernel, gpflow.kernels.base.ReducingCombination):
        lengthscales = []
        for k in kernel.kernels:
            if isinstance(k, MaskedMarkedSE):
                lengthscales.append(tf.cast(k.get_mark_lengthscale(), dtype=float_type))
        lengthscales = tf.stack(lengthscales)  # dim: (Q_a+Q_o,)
    else:
        raise NotImplementedError('Not implemented for the kernel configuration!')
    return lengthscales


def get_mark_output(kernel, rel_marks, Z_marks):
    # marks should only be taken into account for marked dimensions, other dimensions should return 1
    # Mark and time dimensions can be different (if a non-marked PP exists). One mask can't be used for both.
    mask_not_inf_mark = tf.logical_not(tf.math.is_inf(rel_marks))
    rel_marks_masked = tf.where(mask_not_inf_mark, rel_marks, to_default_float(0.0))
    N = tf.shape(rel_marks_masked)[0]
    M = tf.shape(Z_marks)[0]
    # TODO move to kernel
    if isinstance(kernel, MaskedMarkedSE):
        mark_outputs_list = [kernel.get_mark_covariance(rel_marks_masked, Z_marks)]  # dim: (N, M)
    elif isinstance(kernel, gpflow.kernels.base.ReducingCombination):
        mark_outputs_list = []
        idx_marks = 0
        for _, k in enumerate(kernel.kernels):
            if isinstance(k, MaskedSE):
                # Add ones in place of mark covariance for simplicity in
                mark_outputs_list.append(tf.ones((N, M), dtype=default_float()))
            elif isinstance(k, MaskedMarkedSE):
                mark_outputs_list.append(
                    k.get_mark_covariance(tf.gather(rel_marks_masked, [idx_marks], axis=1),
                                          tf.gather(Z_marks, [idx_marks], axis=1))
                )
                idx_marks += 1
    else:
        mark_outputs_list = [tf.ones((N, M), dtype=default_float())]
    mark_outputs = tf.transpose(tf.stack(mark_outputs_list), (1, 2, 0))  # (N, M, Dt)
    # if single kernel: (N, M, 1)
    return mark_outputs
