# Copyright (C) Secondmind Ltd 2017-2020
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

"""
Prototype Code! This code may not be fully tested, or in other ways fit-for-purpose.
Use at your own risk!
"""

import numpy as np
import tensorflow as tf
from gpflow.utilities import to_default_float


def tf_calc_Psi_matrix_vec(variances, lengthscales, mask_not_inf_time_float, t1_minus_s, t0_minus_s, s,
                           Z_times, mark_output):
    M = tf.shape(Z_times)[0]  # Dt = Q_a + Q_o (equivalent to D in GPRPP)
    Dt = tf.shape(Z_times)[1]  # Dt = Q_a + Q_o (equivalent to D in GPRPP)
    N = tf.shape(t1_minus_s)[0]
    #
    mask_not_inf_time_full = (tf.reshape(mask_not_inf_time_float, (N, 1, 1, Dt, 1))
                              * tf.reshape(mask_not_inf_time_float, (N, 1, 1, 1, Dt)))
    t1_minus_s1 = tf.reshape(t1_minus_s, (N, 1, 1, Dt, 1))
    t1_minus_s2 = tf.reshape(t1_minus_s, (N, 1, 1, 1, Dt))
    t0_minus_s1 = tf.reshape(t0_minus_s, (N, 1, 1, Dt, 1))
    t0_minus_s2 = tf.reshape(t0_minus_s, (N, 1, 1, 1, Dt))
    s1 = tf.reshape(s, (N, 1, 1, Dt, 1))
    s2 = tf.reshape(s, (N, 1, 1, 1, Dt))
    gamma1 = tf.reshape(variances, (1, 1, 1, Dt, 1))
    gamma2 = tf.reshape(variances, (1, 1, 1, 1, Dt))
    prod_gamma = gamma1 * gamma2
    alpha1_sqrt = tf.reshape(lengthscales, (1, 1, 1, Dt, 1))
    alpha2_sqrt = tf.reshape(lengthscales, (1, 1, 1, 1, Dt))
    alpha1 = tf.square(alpha1_sqrt)
    alpha2 = tf.square(alpha2_sqrt)
    z1 = tf.reshape(Z_times, (1, M, 1, Dt, 1))
    z2 = tf.reshape(Z_times, (1, 1, M, 1, Dt))
    # Terms
    # root sum square of lengthscales, dim: (1, 1, 1, Dt, Dt)
    rss_lengthscales = tf.sqrt(to_default_float(2.0) * (alpha1 + alpha2))
    erf_denom = rss_lengthscales * (alpha1_sqrt * alpha2_sqrt)
    # dim: (1, 1, 1, Dt, Dt)
    mult = tf.sqrt(to_default_float(np.pi)) * alpha1_sqrt * alpha2_sqrt / rss_lengthscales
    exp_arg = - tf.square(z1 + s1 - z2 - s2) / (to_default_float(2.0) * (alpha1 + alpha2))  #
    erf_val = tf.math.erf(
        (alpha1 * (t1_minus_s2 - z2) + alpha2 * (t1_minus_s1 - z1)) / erf_denom
    ) - tf.math.erf(
        (alpha1 * (t0_minus_s2 - z2) + alpha2 * (t0_minus_s1 - z1)) / erf_denom
    )  # (N, M, M, Dt, Dt)
    prod_exp = tf.exp(exp_arg) * mult * erf_val  # dim: (N x M x M x Dt x Dt)
    # sum over N and last two dimensions
    Psi_exp_dims = mask_not_inf_time_full * prod_gamma * prod_exp  # dim: (N x M x M x Dt x Dt)
    if mark_output is not None:
        mark_output_exp_dims = tf.reshape(mark_output, (N, M, 1, Dt, 1)) * tf.reshape(mark_output, (N, 1, M, 1, Dt))
        Psi_exp_dims *= mark_output_exp_dims
    Psi = tf.reduce_sum(Psi_exp_dims, axis=(0, 3, 4))
    return Psi  # returns (M x M) array
