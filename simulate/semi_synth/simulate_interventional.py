import os
import numpy as np
import pandas as pd
import tensorflow as tf

from plot.plot_simulation import plot_path_specific_trajectories
from simulate.semi_synth.sample_multiple import sample_multiple_trajectories
from simulate.semi_synth.train_sampler import get_oracle_sampler


def sample_interventional_trajectories(exp_ids, sample_patient_ids, outcome_time_domains, treatment_time_domains, args):
    # sample_interventional_trajectories(init_args)
    sample_path = os.path.join(args.samples_dir, f'interventional_trajectories.npz')
    figures_path = os.path.join(args.samples_dir, 'figures')
    os.makedirs(figures_path, exist_ok=True)
    #
    if not os.path.exists(sample_path):
        n_exp_id = len(exp_ids)
        models, model_types, model_ids = get_experiment_models(exp_ids, args)
        patient_multiple_datasets, patient_outcomes, algorithm_logs = [], [], []
        for i, pidx in enumerate(sample_patient_ids):
            print(f'Sampling {i}th interventional trajectories for Patient[{pidx}]...')
            actions, outcomes, outcomes_norm, algorithm_log = sample_multiple_trajectories(models, model_types,
                                                                                           pidx, n_exp_id,
                                                                                           outcome_time_domains[i],
                                                                                           treatment_time_domains[i],
                                                                                           args)
            multiple_ds = [(outcome[:, 0], outcome[:, 1], action[:, 0], action[:, 1], outcome_norm[:, 1])
                           for action, outcome, outcome_norm in zip(actions, outcomes, outcomes_norm)]
            patient_multiple_datasets.append(multiple_ds)
            patient_outcomes.append(outcomes)
            algorithm_logs.append(algorithm_log)
        ds_dict = {
            'ds': patient_multiple_datasets,
            'algorithm_log': algorithm_logs,
            'outcomes': patient_outcomes,
        }
        np.savez(sample_path, **ds_dict, allow_pickle=True)
    else:
        ds_dict = np.load(sample_path, allow_pickle=True)

    if args.plot_interventional_trajectories:
        models, model_types, model_ids = get_experiment_models(exp_ids, args)
        patient_multiple_datasets, algorithm_logs = ds_dict['ds'], ds_dict['algorithm_log']
        for i in range(len(patient_multiple_datasets)):
            multiple_ds, algorithm_log = patient_multiple_datasets[i], algorithm_logs[i]
            plot_dict = {
                'oracle': r'oracle',
                'vbpp_schulam': r'M1',
                # 'gprpp_schulam': r'M3',
                # 'gprpp_fpca': r'Z21-1',
                'oracle_fpca': r'Z21-2',
                'h23': r'H22',
                'interventional': r'our',
            }
            plot_path_specific_trajectories(models, model_types, exp_ids, multiple_ds,
                                            algorithm_log, i, figures_path, outcome_time_domains[0],
                                            plot_dict, args)

    return ds_dict


def get_experiment_models(exp_ids, args):
    model_ids = ('do(0,0)', 'do(0,1)', 'do(1,0)', 'do(1,1)')
    models = get_oracle_cf_models(args)
    n_cf = len(models)
    model_types = [('gprpp', 'oracle_gp')] * n_cf
    for exp_id in exp_ids[1:]:
        compared_models = get_estimated_cf_models(exp_id, args)
        estimated_model_types = get_estimated_model_types(exp_id)
        models += compared_models
        model_types += [estimated_model_types] * n_cf
    return models, model_types, model_ids


def report_metrics(exp_ids, patient_outcomes, args):
    mse_y00, mse_y11 = {ei: [] for ei in exp_ids}, {ei: [] for ei in exp_ids}
    mse_y01, mse_y10 = {ei: [] for ei in exp_ids}, {ei: [] for ei in exp_ids}
    mse_nde_01, mse_nie_01 = {ei: [] for ei in exp_ids}, {ei: [] for ei in exp_ids}
    mse_nde_10, mse_nie_10 = {ei: [] for ei in exp_ids}, {ei: [] for ei in exp_ids}
    mse_te = {ei: [] for ei in exp_ids}
    for pidx, outcomes in enumerate(patient_outcomes):
        # oracle
        no = len(outcomes[0][:, 0])
        y00, y11 = outcomes[0][:, 1], outcomes[3][:, 1]
        y01, y10 = outcomes[1][:, 1], outcomes[2][:, 1]
        nde_01, nie_01 = y11 - y01, y01 - y00
        nde_10, nie_10 = y10 - y00, y11 - y10
        te = y11 - y00
        #
        # estimations
        for i in range(len(exp_ids)):
            exp_id = exp_ids[i]
            y00_i, y11_i = outcomes[i*4][:, 1], outcomes[i*4+3][:, 1]
            y01_i, y10_i = outcomes[i*4+1][:, 1], outcomes[i*4+2][:, 1]
            mse_y00[exp_id].append(np.mean(np.square(y00 - y00_i)))
            mse_y01[exp_id].append(np.mean(np.square(y01 - y01_i)))
            mse_y10[exp_id].append(np.mean(np.square(y10 - y10_i)))
            mse_y11[exp_id].append(np.mean(np.square(y11 - y11_i)))
            #
            nde_01_i, nie_01_i = y11_i - y01_i, y01_i - y00_i
            nde_10_i, nie_10_i = y10_i - y00_i, y11_i - y10_i
            te_i = y11_i - y00_i
            mse_nde_01[exp_id].append(np.mean(np.square(nde_01 - nde_01_i)))
            mse_nie_01[exp_id].append(np.mean(np.square(nie_01 - nie_01_i)))
            mse_nde_10[exp_id].append(np.mean(np.square(nde_10 - nde_10_i)))
            mse_nie_10[exp_id].append(np.mean(np.square(nie_10 - nie_10_i)))
            mse_te[exp_id].append(np.mean(np.square(te - te_i)))

    for m, n in zip([mse_y00, mse_y01, mse_y10, mse_y11, mse_nde_01, mse_nie_01, mse_nde_10, mse_nie_10, mse_te],
                    ['mse_y00', 'mse_y01', 'mse_y10', 'mse_y11',
                     'mse_nde_01', 'mse_nie_01', 'mse_nde_10', 'mse_nie_10', 'mse_te']):
        df = pd.DataFrame.from_dict(m)
        print(n)
        print(df.mean())
        df.to_csv(os.path.join(args.output_dir, f'{n}.csv'))


def get_oracle_cf_models(args):
    models_int = [get_oracle_sampler(period, args) for period in args.periods]  # [do(0,0), do(1,1)]
    models_cf_ = [get_oracle_sampler(period, args) for period in args.periods]  # to create [do(0,1), do(1,0)]
    models_cf = [  # [do(0,0), do(0,1), do(1,0), do(1,1)]
        models_int[0],  # do(0,0)
        (models_cf_[1][0], models_cf_[0][1]),  # do(0,1)
        (models_cf_[0][0], models_cf_[1][1]),  # do(1,0)
        models_int[1]  # do(1,1)
    ]
    return models_cf


def get_estimated_cf_models(exp_id, args):
    models_int = get_estimated_model(exp_id, args)  # [do(0,0), do(1,1)]
    models_cf_ = get_estimated_model(exp_id, args)  # to create [do(0,1), do(1,0)]
    models_cf = [  # [do(0,0), do(0,1), do(1,0), do(1,1)]
        models_int[0],  # do(0,0)
        (models_cf_[1][0], models_cf_[0][1]),  # do(0,1)
        (models_cf_[0][0], models_cf_[1][1]),  # do(1,0)
        models_int[1]  # do(1,1)
    ]
    return models_cf


def get_estimated_model_types(exp_name):
    if exp_name == 'vbpp_schulam':
        estimated_model_types = ('vbpp', 'schulam')
    elif exp_name == 'gprpp_schulam':
        estimated_model_types = ('gprpp', 'schulam')
    elif exp_name == 'gprpp_fpca':
        estimated_model_types = ('gprpp', 'fpca')
    elif exp_name == 'gprpp_hua':
        estimated_model_types = ('gprpp', 'hua')
    elif exp_name == 'vbpp':
        estimated_model_types = ('vbpp', 'estimated_gp')
    elif exp_name == 'vbpp_fpca':
        estimated_model_types = ('vbpp', 'fpca')
    elif exp_name == 'oracle_fpca':
        estimated_model_types = ('oracle', 'fpca')
    elif exp_name == 'hua':
        estimated_model_types = ('hua', 'hua')
    elif exp_name == 'hua_nptr':
        estimated_model_types = ('hua', 'estimated_gp')
    elif exp_name in ['observational', 'interventional', 'counterfactual', 'h23']:
        estimated_model_types = ('gprpp', 'estimated_gp')
    else:
        raise ValueError('Experiment does not exist!')
    return estimated_model_types


def get_estimated_model(exp_name, args):
    if exp_name == 'vbpp_schulam':
        estimated_model = [
            (
                (tf.saved_model.load(os.path.join(args.treatment_model_dirs[period], 'vbpp')),
                 tf.saved_model.load(os.path.join(args.treatment_model_dirs[period], 'mark_intensity'))),
                args.outcome_models_schulam[period]
            )
            for period in args.periods
        ]
    elif exp_name == 'vbbp_fpca':
        estimated_model = [
            (
                (tf.saved_model.load(os.path.join(args.treatment_model_dirs[period], 'vbpp')),
                 tf.saved_model.load(os.path.join(args.treatment_model_dirs[period], 'mark_intensity'))),
                args.outcome_models_fpca[period]
            )
            for period in args.periods
        ]
    elif exp_name == 'oracle_fpca':
        estimated_model = [
            (
                get_oracle_sampler(period, args)[0],
                args.outcome_models_fpca[period]
            )
            for period in args.periods
        ]
    elif exp_name == 'gprpp_fpca':
        estimated_model = [
            (
                (tf.saved_model.load(os.path.join(args.treatment_model_dirs[period], 'time_intensity')),
                 tf.saved_model.load(os.path.join(args.treatment_model_dirs[period], 'mark_intensity'))),
                args.outcome_models_fpca[period]
            )
            for period in args.periods
        ]
    elif exp_name == 'gprpp_schulam':
        estimated_model = [
            (
                (tf.saved_model.load(os.path.join(args.treatment_model_dirs[period], 'time_intensity')),
                 tf.saved_model.load(os.path.join(args.treatment_model_dirs[period], 'mark_intensity'))),
                args.outcome_models_schulam[period]
            )
            for period in args.periods
        ]
    elif exp_name == 'gprpp_hua':
        estimated_model = [
            (
                (tf.saved_model.load(os.path.join(args.treatment_model_dirs[period], 'time_intensity')),
                 tf.saved_model.load(os.path.join(args.treatment_model_dirs[period], 'mark_intensity'))),
                args.outcome_models_hua[period]
            )
            for period in args.periods
        ]
    elif exp_name == 'vbpp':
        estimated_model = [
            (
                (tf.saved_model.load(os.path.join(args.treatment_model_dirs[period], 'vbpp')),
                 tf.saved_model.load(os.path.join(args.treatment_model_dirs[period], 'mark_intensity'))),
                tf.saved_model.load(os.path.join(args.outcome_model_dirs[period], 'outcome_model'))
            )
            for period in args.periods
        ]
    elif exp_name == 'hua':
        estimated_model = [
            (
                (args.treatment_models_hua[period],
                 tf.saved_model.load(os.path.join(args.treatment_model_dirs[period], 'mark_intensity'))),
                args.outcome_models_hua[period]
            )
            for period in args.periods
        ]
    elif exp_name == 'hua_nptr':
        estimated_model = [
            (
                (args.treatment_models_hua[period],
                 tf.saved_model.load(os.path.join(args.treatment_model_dirs[period], 'mark_intensity'))),
                tf.saved_model.load(os.path.join(args.outcome_model_dirs[period], 'outcome_model'))
            )
            for period in args.periods
        ]
    elif exp_name in ['h23']:
        period_saved = args.periods[0]
        estimated_model = [
            (
                (tf.saved_model.load(os.path.join(args.treatment_model_dirs[period], 'time_intensity')),
                 tf.saved_model.load(os.path.join(args.treatment_model_dirs[period], 'mark_intensity'))),
                tf.saved_model.load(os.path.join(args.outcome_model_h23_dirs[period_saved], 'outcome_model_h23'))
            )
            for period in args.periods
        ]
    elif exp_name in ['observational', 'interventional', 'counterfactual']:
        estimated_model = [
            (
                (tf.saved_model.load(os.path.join(args.treatment_model_dirs[period], 'time_intensity')),
                 tf.saved_model.load(os.path.join(args.treatment_model_dirs[period], 'mark_intensity'))),
                tf.saved_model.load(os.path.join(args.outcome_model_dirs[period], 'outcome_model'))
            )
            for period in args.periods
        ]
    else:
        raise ValueError('Experiment does not exist!')
    return estimated_model
