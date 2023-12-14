# Copyright 2018-2020 The GPflow Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import gpflow.utilities
import numpy as np
import tensorflow as tf
from gpflow import Parameter
from gpflow.kernels import Kernel
from gpflow.kernels.base import ActiveDims
from gpflow.utilities import positive, to_default_float
import tensorflow_probability as tfp


class MarkCoregion(Kernel):

    def __init__(
        self,
        output_dim: int,
        rank: int,
        beta0: float,
        beta1: float,
        use_bias: bool,
        *,
        active_dims: Optional[ActiveDims] = None,
        name: Optional[str] = None,
    ):
        """
        :param output_dim: number of outputs expected (0 <= X < output_dim)
        :param rank: number of degrees of correlation between outputs
        """

        # assert input_dim == 1, "Coregion kernel in 1D only"
        super().__init__(active_dims=active_dims, name=name)

        self.output_dim = output_dim
        self.rank = rank
        self.use_bias = use_bias
        kappa = np.ones(self.output_dim)
        if self.use_bias:
            self.beta0 = Parameter(beta0, transform=positive())
        self.beta1 = Parameter(beta1, transform=positive())
        self.kappa = Parameter(kappa, transform=positive())

    def output_covariance(self, X, X2):
        W = self.beta1 * X
        W2 = self.beta1 * X2
        if self.use_bias:
            W += self.beta0
            W2 += self.beta0
        B = tf.linalg.matmul(W, W2, transpose_b=True)
        if tf.shape(X)[0] == tf.shape(X2)[0] and tf.shape(X)[0] == tf.shape(self.kappa)[0]:
            B += tf.linalg.diag(self.kappa)
        return B

    def output_variance(self, X):
        W = self.beta1 * X
        if self.use_bias:
            W += self.beta0
        B_diag = tf.reduce_sum(tf.square(W), 1)
        if tf.shape(X)[0] == tf.shape(self.kappa)[0]:
            B_diag += self.kappa
        return B_diag

    def K(self, X, X2=None):
        M = tf.cast(X[:, 1], tf.float64)
        if X2 is None:
            M2 = M
        else:
            M2 = tf.cast(X2[:, 1], tf.float64)
        B = self.output_covariance(tf.reshape(tf.unique(M)[0], (-1, 1)), tf.reshape(tf.unique(M2)[0], (-1, 1)))

        X = tf.cast(X[..., 0], tf.int32)
        if X2 is None:
            X2 = X
        else:
            X2 = tf.cast(X2[..., 0], tf.int32)
        return tf.gather(tf.transpose(tf.gather(tf.transpose(B), X2)), X)

    def K_diag(self, X):
        M = tf.cast(X[:, 1], tf.float64)
        B_diag = self.output_variance(tf.reshape(tf.unique(M)[0], (-1, 1)))
        X = tf.cast(X[:, 0], tf.int32)
        return tf.gather(B_diag, X)


class HierarchicalMarkCoregion(Kernel):

    def __init__(
        self,
        output_dim: int,
        rank: int,
        beta0: float,
        beta1: float,
        sigma_raw: float,
        num_patients: int,
        treatment_patient_idx,
        use_bias: bool,
        *,
        active_dims: Optional[ActiveDims] = None,
        name: Optional[str] = None,
    ):
        """
        :param output_dim: number of outputs expected (0 <= X < output_dim)
        :param rank: number of degrees of correlation between outputs
        """

        # assert input_dim == 1, "Coregion kernel in 1D only"
        super().__init__(active_dims=active_dims, name=name)

        self.output_dim = output_dim
        self.rank = rank
        self.use_bias = use_bias
        kappa = np.ones(self.output_dim)
        if self.use_bias:
            self.mu_beta0 = Parameter(beta0, transform=positive())
            self.beta0_raw = Parameter(to_default_float(beta0) * tf.ones(num_patients, dtype=tf.float64),
                                       transform=positive())
            self.beta0_raw.prior = tfp.distributions.HalfNormal(to_default_float(1.0))
        self.mu_beta1 = Parameter(beta1, transform=positive())
        self.beta1_raw = Parameter(to_default_float(beta1) * tf.ones(num_patients, dtype=tf.float64),
                                   transform=positive())
        self.beta1_raw.prior = tfp.distributions.HalfNormal(to_default_float(1.0))
        self.num_patients = num_patients
        self.t_pidx = treatment_patient_idx
        self.t_pidx2 = treatment_patient_idx
        self.sigma_raw = sigma_raw
        self.kappa = Parameter(kappa, transform=positive())

    def set_t_pidx(self, t_pidx):
        self.t_pidx = t_pidx

    def set_t_pidx2(self, t_pidx2):
        self.t_pidx2 = t_pidx2

    def output_covariance(self, X, X2):
        beta1 = self.mu_beta1 + self.sigma_raw * self.beta1_raw  # (num_patients,)
        beta1_vec = tf.reshape(tf.gather(beta1, self.t_pidx), (-1, 1))  # (num_treatments,)
        W = beta1_vec * X
        if self.use_bias:
            beta0 = self.mu_beta0 + self.sigma_raw * self.beta0_raw  # (num_patients,)
            beta0_vec = tf.reshape(tf.gather(beta0, self.t_pidx), (-1, 1))  # (num_treatments,)
            W += beta0_vec

        beta1_vec = tf.reshape(tf.gather(beta1, self.t_pidx2), (-1, 1))  # (num_treatments,)
        W2 = beta1_vec * X2
        if self.use_bias:
            beta0_vec = tf.reshape(tf.gather(beta0, self.t_pidx2), (-1, 1))  # (num_treatments,)
            W2 += beta0_vec
        B = tf.linalg.matmul(tf.reshape(W, (-1, 1)), tf.reshape(W2, (-1, 1)), transpose_b=True)
        if tf.shape(X)[0] == tf.shape(X2)[0] and tf.shape(X)[0] == tf.shape(self.kappa)[0]:
            B = B + tf.linalg.diag(self.kappa)
        return B

    def output_variance(self, X):
        beta1 = self.mu_beta1 + self.sigma_raw * self.beta1_raw  # (num_patients,)
        beta1_vec = tf.gather(beta1, self.t_pidx)  # (num_treatments,)
        W = beta1_vec * X
        if self.use_bias:
            beta0 = self.mu_beta0 + self.sigma_raw * self.beta0_raw  # (num_patients,)
            beta0_vec = tf.gather(beta0, self.t_pidx)  # (num_treatments,)
            W += beta0_vec
        B_diag = tf.reduce_sum(tf.square(W), 1)
        if tf.shape(X)[0] == tf.shape(self.kappa)[0]:
            B_diag += self.kappa
        return B_diag

    def K(self, X, X2=None):
        M = tf.cast(X[:, 1], tf.float64)
        if X2 is None:
            M2 = M
        else:
            M2 = tf.cast(X2[:, 1], tf.float64)
        B = self.output_covariance(tf.reshape(tf.unique(M)[0], (-1, 1)),
                                   tf.reshape(tf.unique(M2)[0], (-1, 1)),)
        X = tf.cast(X[..., 0], tf.int32)
        if X2 is None:
            X2 = X
        else:
            X2 = tf.cast(X2[..., 0], tf.int32)
        return tf.gather(tf.transpose(tf.gather(tf.transpose(B), X2)), X)

    def K_diag(self, X):
        M = tf.cast(X[:, 1], tf.float64)
        B_diag = self.output_variance(tf.reshape(tf.unique(M)[0], (-1, 1)))
        X = tf.cast(X[:, 0], tf.int32)
        return tf.gather(B_diag, X)
