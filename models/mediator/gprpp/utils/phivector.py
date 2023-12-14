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


import numpy as np
import tensorflow as tf
from gpflow.utilities import to_default_float


def tf_calc_Phi_SqExp_vec(variances, lengthscales, mask_not_inf_time_float, t1_minus_s, t0_minus_s,
                          Z_times, mark_output):
    M = tf.shape(Z_times)[0]  # Dt = Q_a + Q_o (equivalent to D in GPRPP)
    Dt = tf.shape(Z_times)[1]  # Dt = Q_a + Q_o (equivalent to D in GPRPP)
    N = tf.shape(t1_minus_s)[0]
    #
    t1_minus_s_exp_dims = tf.reshape(t1_minus_s, (N, 1, Dt))
    t0_minus_s_exp_dims = tf.reshape(t0_minus_s, (N, 1, Dt))
    mask_not_inf_time_full = tf.reshape(mask_not_inf_time_float, (N, 1, Dt))
    z0 = tf.reshape(Z_times, (1, M, Dt))
    gamma = tf.reshape(variances, (1, 1, Dt))
    alpha_sqrt = tf.reshape(lengthscales, (1, 1, Dt))
    erf_denom = tf.sqrt(to_default_float(2.0)) * alpha_sqrt
    mult = tf.sqrt(to_default_float(np.pi / 2.0)) * alpha_sqrt * gamma
    erf_val = tf.math.erf(
        (t1_minus_s_exp_dims - z0) / erf_denom
    ) - tf.math.erf(
        (t0_minus_s_exp_dims - z0) / erf_denom
    )  # dim: (N, M x Dt)
    Phi_exp_dims = mask_not_inf_time_full * mult * erf_val  # dim: (N x M x Dt)
    if mark_output is not None:
        Phi_exp_dims *= mark_output
    Phi = tf.reduce_sum(Phi_exp_dims, axis=(0, 2))  # (M, )
    return Phi  # Returns (M, ) array
