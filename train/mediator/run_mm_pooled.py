import argparse
import os.path

import numpy as np
import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.models import GPR
from gpflow.utilities import to_default_float

from plot.plot_tm import plot_gprpp_results, plot_fm_pred, compare_gprpp_train_fits, \
    compare_next_action_times, compare_action_marks
from models.mediator.gprpp.utils.tm_utils import get_relative_input_joint, set_time_domain, set_mark_domain, \
    prepare_tm_output_dir
from models.mediator.gprpp.kernel import MaskedSE, MaskedMarkedSE
from models.mediator.gprpp.model import GPRPP
from utils.data_utils import get_updated_meta_df, get_joint_df, get_data_slice, get_mediator_ds


def train_tm_pooled(args):
    # Prepare data
    datasets = get_treatment_datasets(args.period, args.data_slice, args)
    time_model_path = os.path.join(args.patient_model_dir, 'time_intensity')
    action_times, action_marks, outcome_tuples, baseline_times = prepare_tm_input(datasets, args)
    outcome_tuples = [np.stack([oi[:, 0], oi[:, 1] - np.median(oi[:, 1])]).T for oi in outcome_tuples]
    actions_train, actions_test = action_times[:args.n_day_train], action_times[args.n_day_train:]
    print(f'Average meals per day: {np.mean([len(ai) for ai in actions_train]).item()}')
    outcome_tuples_train, outcome_tuples_test = outcome_tuples[:args.n_day_train], outcome_tuples[args.n_day_train:]
    baseline_times_train, baseline_times_test = baseline_times[:args.n_day_train], baseline_times[args.n_day_train:]
    rel_at_actions, rel_at_all_points, abs_all_points = get_relative_input_joint(actions_train,
                                                                                 outcome_tuples_train,
                                                                                 baseline_times_train, args,
                                                                                 D=args.D)
    domain = np.array(args.domain, float).reshape(1, 2)
    args = set_time_domain(rel_at_all_points, args)
    if args.marked:
        marks = np.concatenate([marked[:, -1] for marked in rel_at_all_points])
        args = set_mark_domain(marks, args)
    print(f'Time domain: {args.treatment_time_domain}')
    print(f'Mark domain: {args.mark_domain}')
    evs_start = []
    for action, outcome_tuple in zip(actions_train, outcome_tuples_train):
        ev_start = []
        if 'b' in args.tm_components:
            ev_start.append(baseline_times[0].item())
        if 'a' in args.tm_components:
            ev_start.append(action[0])
        if 'o' in args.tm_components:
            ev_start.append(outcome_tuple[0, 0])
        evs_start.append(np.array(ev_start))
    for ev_start in evs_start:
        assert ev_start.shape[0] == args.Dt

    if not os.path.exists(time_model_path):
        time_model = run_treatment_time_intensity(abs_all_points, rel_at_actions, rel_at_all_points, evs_start,
                                                  domain, args)
        plot_gprpp_results(baseline_times, action_times, outcome_tuples, action_model=time_model,
                           model_figures_dir=args.model_figures_dir, args=args, oracle_model=None)
        save_gprpp_model(time_model, time_model_path, args)
    time_intensity = tf.saved_model.load(time_model_path)
    # plot_next_action(time_intensity, actions, args)

    mark_model_path = os.path.join(args.patient_model_dir, 'mark_intensity')
    if not os.path.exists(mark_model_path):
        mark_model = run_treatment_mark_intensity_pooled(datasets, args)
        save_gpr_model(mark_model, mark_model_path)
    mark_intensity = tf.saved_model.load(mark_model_path)
    return time_intensity, mark_intensity, action_times, action_marks


def get_treatment_datasets(period, data_slice, args):
    df_meta = get_updated_meta_df()
    df_joint = get_joint_df(period)
    datasets = []
    for pidx in args.patient_ids:
        df = get_data_slice(df_meta, df_joint, pidx, period, data_slice)
        single_ds = get_mediator_ds(df, pidx, period, args.use_time_corrections,
                                    args.log_meals, args.meal_threshold_per_day, args.tc_folder)
        datasets.append(single_ds)
    return datasets


def run_treatment_mark_intensity_pooled(ds, args):
    t_day = np.concatenate([ds_patient[2] % 24 for ds_patient in ds])
    m_day = np.concatenate([ds_patient[3] for ds_patient in ds])
    y_mean, y_std = np.mean(m_day), np.std(m_day)
    X = t_day.astype(np.float64).reshape(-1, 1)
    Y = m_day.astype(np.float64).reshape(-1, 1)
    # Y = (Y-y_mean) / y_std
    var_init = 1.0
    kb = gpflow.kernels.Matern12(variance=var_init, lengthscales=1.0)
    # kb.variance.prior = tfp.distributions.HalfNormal(to_default_float(var_init))
    # kb.lengthscales.prior = tfp.distributions.Gamma(to_default_float(5.0), to_default_float(1.0))
    kb.variance.prior = tfp.distributions.HalfNormal(to_default_float(var_init))
    kb.lengthscales.prior = tfp.distributions.HalfNormal(to_default_float(1.0))
    model = gpflow.models.GPR(data=(X, Y), kernel=kb, noise_variance=1.0)
    gpflow.utilities.print_summary(model)
    min_logs = gpflow.optimizers.Scipy().minimize(model.training_loss, model.trainable_variables,
                                                  compile=True, options={"disp": False, "maxiter": 2000})
    gpflow.utilities.print_summary(model)
    Xnew = np.linspace(0.0, 24.0, 50).astype(np.float64).reshape(-1, 1)
    f_mean, f_var = model.predict_f(Xnew)
    f_mean, f_var = f_mean.numpy(), f_var.numpy()
    plot_fm_pred(t_day, m_day, Xnew, f_mean, f_var,
                 path=os.path.join(args.model_figures_dir, f'f_mark_p{args.period}_log{args.log_meals}.pdf'))
    y_mean, y_var = model.predict_y(Xnew)
    y_mean, y_var = y_mean.numpy(), y_var.numpy()
    plot_fm_pred(t_day, m_day, Xnew, y_mean, y_var,
                 path=os.path.join(args.model_figures_dir, f'y_mark_p{args.period}_log{args.log_meals}.pdf'))
    if args.log_meals:
        m_day = np.exp(m_day)
        f_mean, f_var = np.exp(f_mean), (np.exp(f_var) - 1) * np.exp(2 * f_mean + f_var)
        plot_fm_pred(t_day, m_day, Xnew, f_mean, f_var,
                     path=os.path.join(args.model_figures_dir, f'f_mark_p{args.period}_logFalse.pdf'))
        y_mean, y_var = np.exp(y_mean), (np.exp(y_var) - 1) * np.exp(2 * y_mean + y_var)
        plot_fm_pred(t_day, m_day, Xnew, y_mean, y_var,
                     path=os.path.join(args.model_figures_dir, f'y_mark_p{args.period}_logFalse.pdf'))
        print(np.mean(f_mean), np.mean(m_day))
    # f_mean = (f_mean * y_std) + y_mean

    return model


def run_treatment_time_intensity(abs_all_points, rel_at_actions, rel_at_all_points, evs_start, domain, args):
    kernel = build_gprpp_kernel(args)
    Z = build_gprpp_inducing_var(args)
    gpflow.utilities.print_summary(kernel)
    model = build_gprpp_model(args, domain=domain, kernel=kernel, Z=Z)

    def objective_closure():
        return -model.elbo(abs_all_points, rel_at_actions, rel_at_all_points, evs_start)

    min_logs = gpflow.optimizers.Scipy().minimize(objective_closure, model.trainable_variables,
                                                  compile=True,
                                                  options={"disp": True,
                                                           "maxiter": args.maxiter})
    return model


def prepare_tm_input(ds, args):
    action_times, action_marks = [], []
    outcome_tuples = []
    baseline_times = []
    for ds_patient in ds:
        x, y, t, m = ds_patient
        D = np.max(t) // 24
        for d in range(int(D+1)):
            mask_t = np.logical_and(d * 24.0 < t, t < (d+1) * 24.0)
            t_day = t[mask_t] - d * 24.0
            m_day = m[mask_t]
            sampling_offset = t_day[0] - 0.5 if args.remove_night_time else 0.0
            mask_x = np.logical_and(d * 24.0 + sampling_offset < x, x < (d+1) * 24.0)
            x_day = x[mask_x] - d * 24.0
            y_day = y[mask_x]

            if len(t_day) > 0:
                action_times.append(t_day)
                action_marks.append(m_day)
                outcome_tuples.append(np.stack([x_day, y_day]).T)
                baseline_times.append(np.zeros((1, 1)))

    return action_times, action_marks, outcome_tuples, baseline_times


def build_gprpp_inducing_var(args):
    Z_array = []
    if 'b' in args.tm_components:
        Z_array.append(np.linspace(0.0, 24.0, args.M_times))
    if 'a' in args.tm_components:
        Z_array.append(np.linspace(*args.treatment_time_domain, args.M_times))
    if 'o' in args.tm_components:
        Z_array.append(np.ones(args.M_times) * 12.0)
        Z_array.append(np.linspace(*args.mark_domain, args.M_times))
    Z = np.stack(Z_array).T
    return Z


def build_gprpp_kernel(args):
    Dt, marked_dt = args.Dt, args.marked_dt
    share_variance, share_lengthscales = args.share_variance, args.share_lengthscales
    variance_init, variance_prior = args.variance_init, args.variance_prior
    lengthscales_init = args.lengthscales_init
    lengthscales_prior = args.lengthscales_prior
    optimize_variance, optimize_lengthscales = args.optimize_variance, args.optimize_lengthscales
    #
    kernel = []
    d = 0
    # Baseline-dependent Kernel
    if 'b' in args.tm_components:
        k0 = MaskedSE(variance=variance_init[0], lengthscales=lengthscales_init[0], active_dims=[0])
        prepare_variable_for_optim(k0.variance, variance_prior, optimize_variance)
        prepare_variable_for_optim(k0.lengthscales, lengthscales_prior, optimize_lengthscales)
        kernel = k0
        d += 1
    # Action-dependent Kernel
    if 'a' in args.tm_components:
        kd = MaskedSE(variance=variance_init[d],
                      lengthscales=lengthscales_init[d], active_dims=[d])
        prepare_variable_for_optim(kd.variance, variance_prior, optimize_variance)
        prepare_variable_for_optim(kd.lengthscales, lengthscales_prior, optimize_lengthscales)
        d += 1
        kernel = kernel + kd if kernel else kd
    # Outcome-dependent Kernel
    if 'o' in args.tm_components:
        kd = MaskedMarkedSE(variance=variance_init[d],
                            lengthscales=[lengthscales_init[d], lengthscales_init[d+1]], active_dims=[d, d+1])
        prepare_variable_for_optim(kd.variance, variance_prior, optimize_variance)
        prepare_variable_for_optim(kd.lengthscales, lengthscales_prior, optimize_lengthscales)
        d += 1
        kernel = kernel + kd if kernel else kd

    return kernel


def prepare_variable_for_optim(variable, prior_value, optimize_variable):
    if optimize_variable:
        variable.prior = tfp.distributions.HalfNormal(to_default_float(prior_value))
    else:
        gpflow.utilities.set_trainable(variable, False)


def save_gprpp_model(model: GPRPP, model_path, args):
    model.predict_lambda_compiled = tf.function(
        model.predict_lambda,
        input_signature=[tf.TensorSpec(shape=[None, args.D], dtype=tf.float64)]
    )
    model.predict_f_compiled = tf.function(
        model.predict_f,
        input_signature=[tf.TensorSpec(shape=[None, args.D], dtype=tf.float64)]
    )
    model.predict_integral_term_compiled = tf.function(
        model.predict_integral_term,
        input_signature=[
            [tf.TensorSpec(shape=[None, 1], dtype=tf.float64)],
            [tf.TensorSpec(shape=[None, args.D], dtype=tf.float64)],
            [tf.TensorSpec(shape=[3], dtype=tf.float64)],
        ]
    )
    tf.saved_model.save(model, model_path)


def save_gpr_model(model: GPR, model_path):
    model.predict_f_compiled = tf.function(
        lambda xnew: model.predict_f(xnew, full_cov=False),
        input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)]
    )
    model.predict_y_compiled = tf.function(
        lambda xnew: model.predict_f(xnew, full_cov=False),
        input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)]
    )
    model.predict_y_compiled_full_cov = tf.function(
        lambda xnew: model.predict_f(xnew, full_cov=True),
        input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)]
    )
    model.predict_f_samples_compiled = tf.function(
        model.predict_f_samples,
        input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)]
    )
    tf.saved_model.save(model, model_path)


def build_gprpp_model(args, domain, kernel, Z):
    beta0, marked = args.beta0, args.marked
    M = args.M_times
    time_dims, mark_dims = args.time_dims, args.mark_dims
    #
    inducing_points = gpflow.inducing_variables.InducingPoints(Z)
    q_mu = np.zeros(M)
    q_S = np.eye(M)
    model = GPRPP(inducing_points, kernel, domain, q_mu, q_S, beta0=beta0,
                  time_dims=time_dims, mark_dims=mark_dims, marked=marked)
    gpflow.utilities.set_trainable(model.inducing_variable.Z, False)
    gpflow.utilities.set_trainable(model.beta0, False)
    gpflow.utilities.print_summary(model)
    return model


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='models.treatment')
    parser.add_argument('--n_day_train', type=int, default=5)
    parser.add_argument('--Dt', type=int, default=2)  # Number of time dimensions, default=2 (1 tre. + 1 out.)
    parser.add_argument('--M_times', type=int, default=5)
    parser.add_argument('--M_marks', type=int, default=5)
    parser.add_argument('--marked', action='store_true')
    parser.add_argument('--marked_dt', nargs="+", default=[])
    parser.add_argument('--optimize_variance', action='store_true')
    parser.add_argument('--optimize_lengthscales', action='store_true')
    parser.add_argument('--variance_init', nargs="+", default=[1.0])
    parser.add_argument('--lengthscales_init', nargs="+", default=[1.0])
    parser.add_argument('--mark_lengthscales_init', nargs="+", default=[1.0])
    parser.add_argument('--variance_prior', nargs="+", default=[1.0])
    parser.add_argument('--lengthscales_prior', nargs="+", default=[1.0])
    parser.add_argument('--mark_lengthscales_prior', nargs="+", default=[1.0])
    parser.add_argument('--domain', nargs="+", default=[0.0, 24.0])
    parser.add_argument('--beta0', type=float, default=np.sqrt(0.01))
    parser.add_argument('--share_variance', action='store_true')
    parser.add_argument('--share_lengthscales', action='store_true')
    parser.add_argument("--patient_ids", type=str, default='10')
    parser.add_argument('--preprocess_treatments', action='store_true')
    parser.add_argument('--sampling_rate', type=int, default=2)
    parser.add_argument('--sampling_offset', type=float, default=7.2)
    parser.add_argument('--tm_components', type=str, default='bao')
    parser.add_argument('--baseline_dim', type=int, default=0)
    parser.add_argument('--treatment_dim', type=int, default=1)
    parser.add_argument('--outcome_dim', type=int, default=2)
    parser.add_argument('--log_meals', action='store_true')
    parser.add_argument('--remove_night_time', action='store_true')
    parser.add_argument('--use_time_corrections', action='store_true')
    parser.add_argument("--tc_folder", type=str, )
    parser.add_argument('--meal_threshold_per_day', type=int, default=0)
    #
    parser.add_argument("--period", type=str, default='Baseline')
    parser.add_argument("--data_slice", type=int, default=1)
    parser.add_argument("--save_sampler", action='store_true')
    parser.add_argument("--sampler_dir", type=str, default='models.sampler')
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    #
    init_args = parser.parse_args()
    init_args.patient_ids = [int(s) for s in init_args.patient_ids.split(',')]
    init_args.domain = [float(n) for n in init_args.domain]
    init_args.variance_init = [float(n) for n in init_args.variance_init]
    init_args.variance_prior = [float(n) for n in init_args.variance_prior]
    init_args.lengthscales_init = [float(n) for n in init_args.lengthscales_init]
    init_args.lengthscales_prior = [float(n) for n in init_args.lengthscales_prior]
    init_args.mark_lengthscales_init = [float(n) for n in init_args.mark_lengthscales_init]
    init_args.mark_lengthscales_prior = [float(n) for n in init_args.mark_lengthscales_prior]
    init_args.marked_dt = [int(n) - 1 for n in init_args.marked_dt]
    init_args.D = init_args.Dt + len(init_args.marked_dt)
    init_args.time_dims = list(range(init_args.Dt))
    init_args.mark_dims = list(range(init_args.Dt, init_args.D))
    if init_args.period == 'Compare':
        patient_time_models, patient_mark_models, action_times_train, marks_train = [], [], [], []
        time_domains, mark_domains = [], []
        for period_, data_slice_ in zip(['Baseline', 'Operation'], [1, 2]):
            init_args.period = period_
            init_args.data_slice = data_slice_
            init_args = prepare_tm_output_dir(init_args, is_multiple=True)
            time_model_period, mark_model_period, action_times_period, marks_period = train_tm_pooled(init_args)
            treatment_time_domain = (init_args.treatment_time_domain[0], init_args.treatment_time_domain[1])
            treatment_mark_domain = (init_args.mark_domain[0], init_args.mark_domain[1])
            time_domains.append(treatment_time_domain)
            mark_domains.append(treatment_mark_domain)
            patient_time_models.append(time_model_period)
            patient_mark_models.append(mark_model_period)
            action_times_train.append(action_times_period)
            marks_train.append(marks_period)
        init_args.period = 'Compare'
        init_args = prepare_tm_output_dir(init_args, is_multiple=True)
        init_args.treatment_time_domain = (np.min([td[0] for td in time_domains]),
                                           np.max([td[1] for td in time_domains]))
        init_args.mark_domain = (np.min([md[0] for md in mark_domains]),
                                           np.max([md[1] for md in mark_domains]))
        compare_gprpp_train_fits(patient_time_models, patient_mark_models, args=init_args)
        # compare_fm_pred(patient_mark_models, args=init_args)
        np.random.seed(init_args.seed)
        tf.random.set_seed(init_args.seed)
        compare_next_action_times(patient_time_models, action_times_train, args=init_args)
        compare_action_marks(patient_mark_models, action_times_train, marks_train, args=init_args)
    else:
        init_args = prepare_tm_output_dir(init_args, is_multiple=True)
        time_model_period, mark_model_period, action_times_period, marks_period = train_tm_pooled(init_args)
