import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import gpflow
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.utilities import to_default_float


from plot.plot_tm import plot_vbpp
from models.mediator.vbpp import VBPP
from models.mediator.vbpp.model import integrate_log_fn_sqr


def build_glucose_action_model(events, domain, M=20):
    kernel = gpflow.kernels.SquaredExponential(variance=1.0, lengthscales=5.0)
    kernel.variance.prior = tfp.distributions.HalfNormal(to_default_float(1.0))
    kernel.lengthscales.prior = tfp.distributions.HalfNormal(to_default_float(5.0))
    Z = np.linspace(domain.min(axis=1), domain.max(axis=1), M)
    feature = gpflow.inducing_variables.InducingPoints(Z)
    q_mu = np.zeros(M)
    q_S = np.eye(M)
    model = VBPP(feature, kernel, domain, q_mu, q_S, beta0=0.1,
                 num_observations=len(events))
    gpflow.utilities.set_trainable(model.inducing_variable.Z, False)
    gpflow.utilities.set_trainable(model.beta0, False)
    return model


def train_vbpp(events, args):
    def objective_closure():
        return -model.elbo(events)

    domain = np.array(args.domain).reshape(1, 2)
    model = build_glucose_action_model(events, domain)
    gpflow.optimizers.Scipy().minimize(objective_closure, model.trainable_variables,
                                       compile=True,
                                       options={"disp": False,
                                                "maxiter": args.maxiter}
                                       )
    gpflow.utilities.print_summary(model)
    X, lambda_mean, lower, upper = predict_vbpp(model, domain)
    plot_vbpp(events, X, lambda_mean, upper, lower,
              title=f'Daily Meal Profile of Period={args.period}',
              plot_path=os.path.join(args.model_figures_dir, f'vbpp_pred.pdf'))
    return model


def predict_vbpp_data_term(model, new_events):
    data_term = 0.0
    for evs in new_events:
        if evs.shape[0] > 0:
            mean, var = model.predict_f_compiled(Xnew=evs)
            data_term += tf.reduce_sum(integrate_log_fn_sqr(mean, var))
    return data_term


def predict_vbpp_ll_lbo(model, new_events):
    data_term = predict_vbpp_data_term(model, new_events)
    integral_term = model.predict_integral_term_compiled()
    integral_term = len(new_events) * integral_term
    ll_lbo = data_term + integral_term
    return ll_lbo, data_term, -integral_term


def predict_vbpp(model, domain):
    X = np.linspace(domain.min(axis=1), domain.max(axis=1), 100)
    lambda_mean, lower, upper = model.predict_lambda_and_percentiles(X)
    lower = lower.numpy().flatten()
    upper = upper.numpy().flatten()
    return X, lambda_mean, lower, upper


def save_vbpp_model(model: VBPP, output_path):
    model.predict_lambda_compiled = tf.function(
        model.predict_lambda,
        input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)]
    )
    model.predict_f_compiled = tf.function(
        model.predict_f,
        input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)]
    )
    model.predict_integral_term_compiled = tf.function(
        model.predict_integral_term,
        input_signature=[]
    )
    tf.saved_model.save(model, output_path)
