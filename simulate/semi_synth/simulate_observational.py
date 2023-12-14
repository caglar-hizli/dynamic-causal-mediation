import argparse
import os
import numpy as np
import tensorflow as tf

from plot.plot_om import plot_fs_pred
from sample_joint import sample_joint_tuples
from simulate.semi_synth.train_sampler import get_oracle_sampler
from utils import constants


def load_patients(args):
    train_data = {}
    for period in args.periods:
        train_data[period] = sample_patients_period(period, args)
    return train_data


def sample_patients_period(period, args):
    samples_period_dir = os.path.join(args.samples_dir, period)
    patient_sample_path = os.path.join(samples_period_dir, f'patients.npz')
    if not os.path.exists(patient_sample_path):
        samples_period_figure_dir = os.path.join(samples_period_dir, 'figure')
        os.makedirs(samples_period_dir, exist_ok=True)
        os.makedirs(samples_period_figure_dir, exist_ok=True)
        ds_period = []
        oracler_sampler = get_oracle_sampler(period, args)
        for pidx in range(args.n_patient):
            outcome_pidx = pidx % len(args.oracle_outcome_pid)
            actions, outcomes, f_means, f_vars, f_median = sample_joint_tuples(oracler_sampler, period, outcome_pidx,
                                                                               n_day=args.n_day, args=args)
            ds_pidx = (outcomes[:, 0], outcomes[:, 1], actions[:, 0], actions[:, 1], f_median)
            print(f'Period[{period}], Patient[{outcome_pidx}], #treatments: {len(actions[:, 0])}')
            ds_period.append(ds_pidx)
            plot_patient_sample(ds_pidx, f_means, f_vars,
                                file_name=os.path.join(samples_period_figure_dir, f'pidx{pidx}-p{period}.pdf'))

        dataset_dict = {'dataset': ds_period}
        np.savez(patient_sample_path, **dataset_dict, allow_pickle=True)

    dataset_dict = np.load(patient_sample_path, allow_pickle=True)
    patient_datasets = dataset_dict['dataset']
    save_r_fpca_format(patient_datasets, period, args)
    return  patient_datasets


def save_r_fpca_format(patient_datasets, period, args):
    treated = 1 if period == 'Operation' else 0
    mediators = []
    for pidx in range(args.n_patient):
        ds = patient_datasets[pidx]
        x, t, m = ds[0], ds[2], ds[3]
        for xi in x:
            mi = 0.0
            effective_mask = np.logical_and(t < xi, np.abs(xi - t) < args.T_treatment)
            mi += np.sum(m[effective_mask])
            mediators.append(mi)
    m = np.array(mediators)
    # Add jitter to time and Normalize Time to [0, 1]
    x = np.concatenate([ds[0] for ds in patient_datasets])
    x += np.random.randn(x.shape[0]) * 0.01
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    y = np.concatenate([ds[1] for ds in patient_datasets])
    treatment = np.ones_like(x) * treated
    y_lengths = [len(ds[1]) for ds in patient_datasets]
    idx = np.arange(1, args.n_patient+1)
    idx_full = np.repeat(idx, y_lengths)
    samples_period_dir = os.path.join(args.samples_dir, period)
    outcome_path = os.path.join(samples_period_dir, f'outcome_r_data_fpca.npz')
    np.savez(outcome_path, ID=idx_full, Treatment=treatment, Time=x, Outcome=y, Mediator=m, allow_pickle=True)


def plot_patient_sample(ds, f_means, f_vars, file_name):
    plot_fs_pred(ds[0], [f_means[-1]], [f_vars[-1]],
                 [r'$\mathbf{f_b}$', r'$\mathbf{f_a}$', r'$\mathbf{f}$'],
                 ['tab:orange', 'tab:green', 'tab:blue'], ds, path=file_name, plot_var=False)


def get_run_id(args):
    pa_str = ','.join([str(pi) for pi in args.oracle_treatment_pid])
    po_str = ','.join([str(pi) for pi in args.oracle_outcome_pid])
    run_id_ = f'np{args.n_patient}.ntr{args.n_day}.pa{pa_str}.po{po_str}' \
              f'.no{args.n_outcome}.s{args.seed}'
    return run_id_


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampler_dir', type=str, default='models.sampler')
    parser.add_argument('--samples_dir', type=str, default='samples')
    parser.add_argument('--oracle_outcome_pid_str', type=str, default='12,28,29')
    parser.add_argument('--n_patient', type=int, default=2)
    parser.add_argument('--n_day', type=int, default=1)
    parser.add_argument('--prob_activity', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--tc_folder', type=str)
    init_args = parser.parse_args()
    init_args.oracle_outcome_pid = [int(pid) for pid in init_args.oracle_outcome_pid_str.split(',')]
    init_dict = vars(init_args)
    init_dict.update(constants.PARAMS_GENERAL)
    init_dict.update(constants.PARAMS_SAMPLING_ALG)
    init_dict.update(constants.GPRPP_PARAMS)
    init_dict.update(constants.GPRPP_FAO)
    init_args = argparse.Namespace(**init_dict)
    #
    np.random.seed(init_args.seed)
    tf.random.set_seed(init_args.seed)
    run_id = get_run_id(init_args)
    init_args.samples_dir = os.path.join(init_args.samples_dir, run_id)
    os.makedirs(init_args.samples_dir, exist_ok=True)
    load_patients(init_args)
