import argparse
import os
import numpy as np
import tensorflow as tf
import gpflow as gpf
import tensorflow_probability as tfp

from gpflow.utilities import to_default_float

from models.outcome.piecewise_se.model_shared_marked import TRSharedMarked
from models.outcome.piecewise_se.model_shared_marked_hierarchical import TRSharedMarkedHier
from utils.data_utils import get_data_slice, train_test_split, get_outcome_ds


def prepare_om_input_marked(ds_trains):
    x = [ds[0].astype(np.float64).reshape(-1, 1) for ds in ds_trains]
    y = [ds[1].reshape(-1, 1) for ds in ds_trains]
    a = [np.hstack([ds[2].astype(np.float64).reshape(-1, 1),
                    ds[3].astype(np.float64).reshape(-1, 1)]) for ds in ds_trains]
    return x, y, a


def get_patient_datasets(df_meta, df_joint, period, data_slice, args):
    ds_trains, ds_alls, ds_tests = [], [], []
    for pid in args.patient_ids:
        print(f'Running Patient[{pid}]')
        df = get_data_slice(df_meta, df_joint, pid, period, data_slice)
        ds = get_outcome_ds(df, pid, period,
                            use_time_corrections=args.use_time_corrections,
                            is_meal_logg=args.log_meals,
                            tc_folder=args.tc_folder)
        ds_train, ds_test = train_test_split(ds, args.n_day_train)
        ds_trains.append(ds_train)
        ds_tests.append(ds_test)
        ds_alls.append(ds)
    return ds_trains, ds_tests, ds_alls


def train_model(model, args):
    gpf.utilities.print_summary(model)
    opt = gpf.optimizers.Scipy()
    min_logs = opt.minimize(model.training_loss,
                            model.trainable_variables,
                            compile=True,
                            options={"disp": True,
                                     "maxiter": args.outcome_maxiter})
    gpf.utilities.print_summary(model)
    return model


def get_baseline_kernel():
    kb_per = get_baseline_kernel_periodic()
    return gpf.kernels.Constant() + kb_per


def get_baseline_kernel_periodic():
    kb_per = gpf.kernels.Periodic(base_kernel=get_se_kernel_periodic(), period=24.0)
    gpf.utilities.set_trainable(kb_per.period, False)
    return kb_per


def get_se_kernel_periodic():
    kb_se = gpf.kernels.SquaredExponential(variance=1.0, lengthscales=10.0)
    kb_se.variance.prior = tfp.distributions.HalfNormal(to_default_float(1.0))
    # kb_se.lengthscales.prior = tfp.distributions.HalfNormal(to_default_float(0.1))
    kb_se.lengthscales.prior = tfp.distributions.Gamma(to_default_float(10.0), to_default_float(1.0))
    return kb_se


def get_treatment_base_kernel():
    kse = gpf.kernels.SquaredExponential(variance=1.0, lengthscales=0.5, active_dims=[0])
    kse.variance.prior = tfp.distributions.HalfNormal(to_default_float(1.0))
    kse.lengthscales.prior = tfp.distributions.Gamma(to_default_float(1.0), to_default_float(2.0))
    return kse


def parse_om_init_args():
    # Target ids:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='models.outcome/single')
    parser.add_argument('--use_time_corrections', action='store_true')
    parser.add_argument('--tc_folder', type=str)
    parser.add_argument('--use_bias', action='store_true')
    parser.add_argument('--log_meals', action='store_true')
    parser.add_argument('--share_baseline', action='store_true')
    parser.add_argument("--patient_ids", type=str, default='12')
    parser.add_argument("--n_day_train", type=int, default=2)
    parser.add_argument("--n_day_test", type=int, default=1)
    parser.add_argument("--period", type=str, default='Compare')
    parser.add_argument("--data_slice", type=int, default=0)
    parser.add_argument("--T_treatment", type=float, default=3)  # in hours
    parser.add_argument("--outcome_maxiter", type=int, default=1000)
    init_args = parser.parse_args()
    init_args.patient_ids = [int(idx) for idx in init_args.patient_ids.split(',')]
    return init_args


def get_om_run_args(period, n_patients, n_day_train, output_dir, maxiter):
    train_params_dict = {
        'output_dir': output_dir,
        'use_time_corrections': True,
        'use_bias': True,
        'log_meals': True,
        'share_baseline': False,
        'patient_ids': list(range(n_patients)),
        'n_day_train': n_day_train,
        'period': period,
        'T_treatment': 3.0,
        'outcome_maxiter': maxiter,
    }
    outcome_args = argparse.Namespace(**train_params_dict)
    return outcome_args


def prepare_om_output_dir(init_args, shared_dir, ordered_pids=False):
    if shared_dir:
        outcome_str = 'outcome.p'
        if ordered_pids:
            outcome_str += f'{init_args.patient_ids[0]}-{init_args.patient_ids[-1]}'
        else:
            outcome_str += ','.join(str(i) for i in init_args.patient_ids)
        init_args.output_dir = os.path.join(init_args.output_dir, init_args.period, outcome_str)
        init_args.output_figures_dir = os.path.join(init_args.output_dir, 'figures')
        os.makedirs(init_args.output_dir, exist_ok=True)
        os.makedirs(init_args.output_figures_dir, exist_ok=True)
    else:
        init_args.patient_dirs = {}
        for pidx in init_args.patient_ids:
            pdir = os.path.join(init_args.output_base_dir, init_args.period, f'outcome.p{pidx}')
            os.makedirs(pdir, exist_ok=True)
            init_args.patient_dirs[pidx] = pdir
    return init_args


def save_model_shared_marked(model: TRSharedMarked, output_dir):
    model.predict_baseline_compiled = tf.function(
        model.predict_baseline,
        input_signature=[[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)] * model.N, ]
    )
    model.predict_ft_w_tnew_compiled = tf.function(
        model.predict_ft_w_tnew,
        input_signature=[[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)] * model.N,
                         [tf.TensorSpec(shape=[None, 2], dtype=tf.float64)] * model.N]
    )
    model.predict_ft_compiled = tf.function(
        model.predict_ft,
        input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64),
                         tf.TensorSpec(shape=[None, 2], dtype=tf.float64)]
    )
    model.log_marginal_likelihood_compiled = tf.function(
        model.log_marginal_likelihood,
        input_signature=[]
    )
    tf.saved_model.save(model, os.path.join(output_dir, f'outcome_model'))


def save_model_shared_marked_hier(model: TRSharedMarkedHier, output_path):
    model.predict_ft_w_tnew_compiled = tf.function(
        model.predict_ft_w_tnew,
        input_signature=[[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)] * model.N,
                         [tf.TensorSpec(shape=[None, 2], dtype=tf.float64)] * model.N,
                         tf.TensorSpec(shape=[None, ], dtype=tf.int32)]
    )
    model.predict_ft_w_tnew_conditional_compiled = tf.function(
        model.predict_ft_w_tnew_conditional,
        input_signature=[[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)] * model.N,
                         [tf.TensorSpec(shape=[None, 1], dtype=tf.float64)] * model.N,
                         [tf.TensorSpec(shape=[None, 2], dtype=tf.float64)] * model.N,
                         tf.TensorSpec(shape=[None, ], dtype=tf.int32),
                         [tf.TensorSpec(shape=[None, 1], dtype=tf.float64)] * model.N,
                         [tf.TensorSpec(shape=[None, 2], dtype=tf.float64)] * model.N,
                         tf.TensorSpec(shape=[None, ], dtype=tf.int32)]
    )
    model.predict_baseline_compiled = tf.function(
        model.predict_baseline,
        input_signature=[[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)] * model.N]
    )
    model.log_marginal_likelihood_compiled = tf.function(
        model.log_marginal_likelihood,
        input_signature=[]
    )
    model.predict_baseline_conditional_compiled = tf.function(
        model.predict_baseline_conditional,
        input_signature=[[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)] * model.N,
                         [tf.TensorSpec(shape=[None, 1], dtype=tf.float64)] * model.N,
                         [tf.TensorSpec(shape=[None, 2], dtype=tf.float64)] * model.N,
                         tf.TensorSpec(shape=[None, ], dtype=tf.int32),
                         [tf.TensorSpec(shape=[None, 1], dtype=tf.float64)] * model.N]
    )
    model.predict_baseline_samples_compiled = tf.function(
        model.predict_baseline_samples,
        input_signature=[[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)] * model.N,
                         tf.TensorSpec(shape=None, dtype=tf.int32)]
    )
    model.predict_ft_single_for_patient_compiled = tf.function(
        model.predict_ft_single_for_patient,
        input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64),
                         tf.TensorSpec(shape=[None, 2], dtype=tf.float64),
                         tf.TensorSpec(shape=None, dtype=tf.int32)]
    )
    tf.saved_model.save(model, output_path)


def save_model_hier_params(model, args):
    beta0 = (model.treatment_kernel.kernels[1].mu_beta0 +
             model.treatment_kernel.kernels[1].sigma_raw * model.treatment_kernel.kernels[1].beta0_raw)
    beta1 = (model.treatment_kernel.kernels[1].mu_beta1 +
             model.treatment_kernel.kernels[1].sigma_raw * model.treatment_kernel.kernels[1].beta1_raw)
    treatment_effect_params = []
    baseline_params = []
    for i, patient_idx in enumerate(args.patient_ids):
        baseline_params.append((model.mean_functions[i].A.numpy().item(), model.mean_functions[i].b.numpy().item()))
        treatment_effect_params.append((beta0[i], beta1[i]))
    baseline_params = np.stack(baseline_params)
    treatment_effect_params = np.stack(treatment_effect_params)
    trained_params = {
        'patient_ids': args.patient_ids,
        'baseline_params': baseline_params,
        'treatment_effect_params': treatment_effect_params
    }
    np.savez(os.path.join(args.output_dir, 'trained_params_dict.npz'), **trained_params)
