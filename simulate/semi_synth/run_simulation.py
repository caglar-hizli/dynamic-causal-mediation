import argparse
import os
import numpy as np
import tensorflow as tf

from plot.plot_simulation import plot_outcome_benchmarks
from utils.om_utils import get_om_run_args, save_model_shared_marked_hier, prepare_om_output_dir
from train.outcome.predict import predict_ft_marked_hier, predict_f_train_fit_shared_marked_hier
from train.outcome.compare_om_shared_marked_hier import run_model
from train.mediator.run_mm_ind import run_treatment_time_intensity, run_treatment_mark_intensity_pooled, save_gprpp_model, \
    save_gpr_model
from train.mediator.run_vbpp import train_vbpp, save_vbpp_model
from models.benchmarks.fpca.model_fpca_from_mcmc import OutcomeModelFpca, load_outcome_params_fpca
from models.benchmarks.schulam.run_schulam_saria import train_schulam
from models.mediator.gprpp.utils.tm_utils import prepare_tm_output_dir, get_gprpp_run_args, get_vbpp_run_args
from simulate.semi_synth.simulate_interventional import sample_interventional_trajectories, report_metrics
from simulate.semi_synth.simulate_observational import load_patients, get_run_id
from simulate.semi_synth.train_sampler import get_oracle_sampler
from utils import constants


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
EXP_IDS_ALL = ['oracle', 'vbpp_schulam', 'vbpp', 'gprpp_schulam', 'gprpp_fpca', 'oracle_fpca', 'h23', 'interventional']
EXP_IDS_HIDDEN_CONFOUNDING = ['oracle', 'gprpp_schulam', 'gprpp_fpca', 'interventional']

def run_synth_exp(args):
    train_data = load_patients(args)
    for period in args.periods:
        args = train_models(train_data, period, args)
    if 'h23' in args.exp_ids:
        train_outcome_model_h23(train_data, args)
    outcome_time_domains = [args.domain for _ in range(args.n_patient)]
    treatment_time_domains = [args.domain for _ in range(args.n_patient)]
    sample_patient_ids = list(range(args.n_patient))
    ds_dict = sample_interventional_trajectories(args.exp_ids, sample_patient_ids,
                                                 outcome_time_domains, treatment_time_domains, args)
    report_metrics(args.exp_ids, ds_dict['outcomes'], args)


def train_models(train_data, period, args):
    oracler_sampler = get_oracle_sampler(period, args)
    ds_train_period = train_data[period]
    train_outcome_model(ds_train_period, period, args)
    train_treatment_model(ds_train_period, period, args, oracler_sampler)
    train_vbpp_treatment_model(ds_train_period, period, args)
    args = train_schulam_outcome_model(ds_train_period, period, args)
    args = get_fpca_outcome_model(ds_train_period, period, args)
    return args


def train_treatment_model(ds_train, period, args, oracle_models):
    tm_run_args = get_gprpp_run_args(period, args.tm_components, args.output_dir)
    tm_run_args = prepare_tm_output_dir(tm_run_args)
    args.treatment_model_dirs[period] = tm_run_args.patient_model_dir
    time_intensity_path = os.path.join(tm_run_args.patient_model_dir, 'time_intensity')
    mark_intensity_path = os.path.join(tm_run_args.patient_model_dir, 'mark_intensity')
    if not os.path.exists(time_intensity_path):
        tm_time = run_treatment_time_intensity(ds_train, tm_run_args, oracle_model=oracle_models[0][0])
        tm_mark = run_treatment_mark_intensity_pooled(ds_train, tm_run_args)
        save_gprpp_model(tm_time, time_intensity_path, tm_run_args)
        save_gpr_model(tm_mark, mark_intensity_path)


def train_vbpp_treatment_model(ds_train, period, args):
    tm_run_args = get_vbpp_run_args(period, args.output_dir)
    tm_run_args = prepare_tm_output_dir(tm_run_args)
    time_intensity_path = os.path.join(tm_run_args.patient_model_dir, 'vbpp')
    if not os.path.exists(time_intensity_path):
        treatment_times = [ds[2].reshape(-1, 1) for ds in ds_train]
        model = train_vbpp(treatment_times, tm_run_args)
        save_vbpp_model(model, time_intensity_path)


def train_schulam_outcome_model(patient_ds, period, args):
    om_run_args = get_om_run_args(period, args.n_patient, args.n_day, args.output_dir, args.outcome_maxiter)
    om_run_args = prepare_om_output_dir(om_run_args, shared_dir=True, ordered_pids=True)
    train_ds_schulam, outcome_model_schulam = train_schulam(patient_ds, high=24.0, args=args)
    args.train_datasets_schulam[period] = train_ds_schulam
    args.outcome_models_schulam[period] = outcome_model_schulam
    for pidx in range(args.n_patient):
        ds = patient_ds[pidx]
        plot_outcome_benchmarks(outcome_model_schulam, 'schulam', ds, period,
                                pidx, om_run_args.output_figures_dir, args, ds_type='train')
    return args


def get_fpca_outcome_model(patient_ds, period, args):
    samples_period_dir = os.path.join(args.samples_dir, period)
    outcome_mcmc_path = os.path.join(samples_period_dir, f'fpca_mcmc_output.npz')
    args.outcome_models_fpca[period] = OutcomeModelFpca(*load_outcome_params_fpca(outcome_mcmc_path))
    om_run_args = get_om_run_args(period, args.n_patient, args.n_day, args.output_dir, args.outcome_maxiter)
    om_run_args = prepare_om_output_dir(om_run_args, shared_dir=True, ordered_pids=True)
    for pidx in range(args.n_patient):
        ds = patient_ds[pidx]
        plot_outcome_benchmarks(args.outcome_models_fpca[period], 'fpca', ds, period,
                                pidx, om_run_args.output_figures_dir, args, ds_type='train')
    return args


def train_outcome_model(ds_trains, period, args):
    n_patient = len(ds_trains)
    om_run_args = get_om_run_args(period, n_patient, args.n_day, args.output_dir, 100)  # TODO
    om_run_args = prepare_om_output_dir(om_run_args, shared_dir=True, ordered_pids=True)
    args.outcome_model_dirs[period] = om_run_args.output_dir
    outcome_model_path = os.path.join(om_run_args.output_dir, 'outcome_model')
    if not os.path.exists(outcome_model_path):
        model = run_model(ds_trains, om_run_args)
        save_model_shared_marked_hier(model, outcome_model_path)
        model = tf.saved_model.load(outcome_model_path)
        predict_ft_marked_hier(model, om_run_args)
        predict_f_train_fit_shared_marked_hier(model, ds_trains, om_run_args.period,
                                               path=om_run_args.output_figures_dir, args=om_run_args)


def train_outcome_model_h23(ds_trains, args):
    per = args.periods[0]  # run only once, and save to the Baseline folder
    om_run_args = get_om_run_args(per, args.n_patient, args.n_day*2+1, args.output_dir, args.outcome_maxiter)
    om_run_args = prepare_om_output_dir(om_run_args, shared_dir=True, ordered_pids=True)
    args.outcome_model_h23_dirs[per] = om_run_args.output_dir
    outcome_model_path = os.path.join(om_run_args.output_dir, 'outcome_model_h23')
    if not os.path.exists(outcome_model_path):
        # concatenate baseline + operation
        ds_concat = []
        ds_bs, ds_op = ds_trains[args.periods[0]], ds_trains[args.periods[1]]
        for ni in range(args.n_patient):
            xb, yb, tb, mb, _ = ds_bs[ni]
            xo, yo, to, mo, _ = ds_op[ni]
            ds_concat.append(
                (np.concatenate([xb, xo + args.hours_day * (args.n_day+1)]),  # leave 1 day in between
                 np.concatenate([yb, yo]),
                 np.concatenate([tb, to + args.hours_day * (args.n_day+1)]),  # leave 1 day in between
                 np.concatenate([mb, mo]))
            )
        model = run_model(ds_concat, om_run_args)
        save_model_shared_marked_hier(model, outcome_model_path)
        model = tf.saved_model.load(outcome_model_path)
        om_run_args.output_figures_dir += '_h23'
        os.makedirs(om_run_args.output_figures_dir, exist_ok=True)
        predict_ft_marked_hier(model, om_run_args)
        predict_f_train_fit_shared_marked_hier(model, ds_concat, om_run_args.period,
                                               path=om_run_args.output_figures_dir, args=om_run_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str)
    parser.add_argument('--sampler_dir', type=str, default='models.sampler')
    parser.add_argument('--samples_dir', type=str, default='samples')
    parser.add_argument('--output_dir', type=str, default='simulation')
    parser.add_argument('--oracle_outcome_pid_str', type=str, default='12,28,29')
    parser.add_argument('--n_patient', type=int, default=10)
    parser.add_argument('--n_day', type=int, default=1)
    parser.add_argument('--prob_activity', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--tc_folder', type=str)
    init_args = parser.parse_args()
    if init_args.task == 'semi-synth':
        init_args.exp_ids = EXP_IDS_ALL
    elif init_args.task == 'hidden-confounding':
        init_args.exp_ids = EXP_IDS_HIDDEN_CONFOUNDING
    else:
        raise ValueError('The task does not exist!')
    #
    init_args.oracle_outcome_pid = [int(pid) for pid in init_args.oracle_outcome_pid_str.split(',')]
    init_dict = vars(init_args)
    init_dict.update(constants.PARAMS_GENERAL)
    init_dict.update(constants.PARAMS_SAMPLING_ALG)
    init_dict.update(constants.GPRPP_PARAMS)
    init_dict.update(constants.GPRPP_FAO)
    init_args = argparse.Namespace(**init_dict)
    #
    run_id = get_run_id(init_args)
    init_args.samples_dir = os.path.join(init_args.samples_dir, run_id)
    init_args.output_dir = os.path.join(init_args.output_dir, run_id)
    init_args.outcome_model_dirs = {}
    init_args.outcome_model_h23_dirs = {}
    init_args.treatment_model_dirs = {}
    init_args.outcome_models_schulam = {}
    init_args.outcome_models_hua = {}
    init_args.treatment_models_hua = {}
    init_args.outcome_models_fpca = {}
    init_args.train_datasets_schulam = {}
    np.random.seed(init_args.seed)
    tf.random.set_seed(init_args.seed)
    run_synth_exp(init_args)
