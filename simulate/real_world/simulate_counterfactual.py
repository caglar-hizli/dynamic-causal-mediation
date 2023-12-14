import argparse
import os
import numpy as np
import tensorflow as tf

from train.mediator.run_mm_pooled import get_treatment_datasets, prepare_tm_input
from simulate.semi_synth.simulate_interventional import sample_interventional_trajectories
from utils import constants


def run_real_world_sampling_task(args):
    exp_ids = ['oracle']
    dataset_patient, dataset_period = {}, {}
    patient_ids_period, patient_ids_array = {per: [] for per in args.periods}, []
    offset_op = 20
    for period, data_slice in zip(args.periods, args.data_slices):
        ds_period = get_treatment_datasets(period, data_slice, args)
        dataset_period[period] = ds_period
        offset = 0 if period == 'Baseline' else offset_op
        for pidx, di in zip(args.patient_ids, ds_period):
            dataset_patient[pidx] = di
            action_times, _, outcome_tuples, _ = prepare_tm_input([di], args)
            patient_ids_array += [args.patient_ids.index(pidx)+offset] * len(outcome_tuples)
            patient_ids_period[period] += [args.patient_ids.index(pidx)] * len(outcome_tuples)
    dataset_array = dataset_period['Baseline'] + dataset_period['Operation']
    action_times, _, outcome_tuples, _ = prepare_tm_input(dataset_array, args)
    args.n_patient = len(outcome_tuples)
    print(args.n_patient, len(patient_ids_array))
    xs = []
    ns = 0
    for pidx in range(args.n_patient):
        deltax = np.mean(outcome_tuples[pidx][:, 0][1:] - outcome_tuples[pidx][:, 0][:-1])
        nx = len(outcome_tuples[pidx][:, 0])
        print(pidx, outcome_tuples[pidx][:, 0].min(), outcome_tuples[pidx][:, 0].max(), nx, deltax)
        ns += nx
        xs.append(deltax)
    print(np.mean(xs), ns)
    print(patient_ids_array)
    print(patient_ids_period)
    outcome_time_domains = [args.domain for _ in range(args.n_patient)]
    treatment_time_domains = [(outcome_tuples[pidx][:, 0].min(), args.domain[-1]) for pidx in range(args.n_patient)]
    # treatment_time_domains = [(18.0, args.domain[-1]) for pidx in range(args.n_patient)]
    sampled_ds_dict = sample_interventional_trajectories(exp_ids, patient_ids_array,
                                                         outcome_time_domains, treatment_time_domains, args)
    sampled_datasets = sampled_ds_dict['ds']

    # REAL-DATA
    ts_hypo_bs, ts_hyper_bs, ts_measured_bs = report_time_spent(dataset_period['Baseline'], 'Baseline',
                                                                args.threshold_hypo, args.threshold_hyper)
    ts_hypo, ts_hyper, ts_measured = report_time_spent(dataset_period['Operation'], 'Operation',
                                                       args.threshold_hypo, args.threshold_hyper)
    # REAL-DATA per patient
    patient_ids_period = {k: np.array(v) for k, v in patient_ids_period.items()}
    patient_ids_array = np.array(patient_ids_array)
    print(len(patient_ids_period['Baseline']), len(patient_ids_period['Operation']))
    print(len(dataset_period['Baseline']), len(dataset_period['Operation']), len(sampled_datasets))
    for i in range(15):
        print(f'-------- PATIENT {i} --------')
        print(f'-------- EMPIRICAL --------')
        report_time_spent([dataset_period['Baseline'][i]], 'Baseline', args.threshold_hypo, args.threshold_hyper)
        report_time_spent([dataset_period['Operation'][i]], 'Operation', args.threshold_hypo, args.threshold_hyper)
        print(f'-------- SAMPLED --------')
        idx_bs = np.where(patient_ids_array == i)[0]
        patient_ds_bs = [sampled_datasets[ii] for ii in idx_bs]
        idx_op = np.where(patient_ids_array == i + offset_op)[0]
        patient_ds_op = [sampled_datasets[ii] for ii in idx_op]
        for p_ds in [patient_ds_bs, patient_ds_op]:
            print(f'------------------------')
            report_time_spent([di[0] for di in p_ds], 'do(0,0)', args.threshold_hypo, args.threshold_hyper)
            report_time_spent([di[1] for di in p_ds], 'do(0,1)', args.threshold_hypo, args.threshold_hyper)
            report_time_spent([di[2] for di in p_ds], 'do(1,0)', args.threshold_hypo, args.threshold_hyper)
            report_time_spent([di[3] for di in p_ds], 'do(1,1)', args.threshold_hypo, args.threshold_hyper)
        # report_time_spent([di[0] for di in patient_ds_op], 'do(0,0)', args.threshold_hypo, args.threshold_hyper)
        # report_time_spent([di[1] for di in patient_ds_op], 'do(0,1)', args.threshold_hypo, args.threshold_hyper)
        # report_time_spent([di[2] for di in patient_ds_op], 'do(1,0)', args.threshold_hypo, args.threshold_hyper)
        # report_time_spent([di[3] for di in patient_ds_op], 'do(1,1)', args.threshold_hypo, args.threshold_hyper)

    # Time spent statistics general
    ts_hypo, ts_hyper, ts_measured = report_time_spent([di[0] for di in sampled_datasets], 'do(0,0)',
                                                       args.threshold_hypo, args.threshold_hyper)
    ts_hypo, ts_hyper, ts_measured = report_time_spent([di[1] for di in sampled_datasets], 'do(0,1)',
                                                       args.threshold_hypo, args.threshold_hyper)
    ts_hypo, ts_hyper, ts_measured = report_time_spent([di[2] for di in sampled_datasets], 'do(1,0)',
                                                       args.threshold_hypo, args.threshold_hyper)
    ts_hypo, ts_hyper, ts_measured = report_time_spent([di[3] for di in sampled_datasets], 'do(1,1)',
                                                       args.threshold_hypo, args.threshold_hyper)
    # Time spent statistics per period
    n_day_bs = len(patient_ids_period['Baseline'])
    print(n_day_bs)
    print('Printing Baseline statistics...')
    sampled_datasets_bs = sampled_datasets[:n_day_bs]
    ts_hypo, ts_hyper, ts_measured = report_time_spent([di[0] for di in sampled_datasets_bs], 'do(0,0)',
                                                       args.threshold_hypo, args.threshold_hyper)
    ts_hypo, ts_hyper, ts_measured = report_time_spent([di[1] for di in sampled_datasets_bs], 'do(0,1)',
                                                       args.threshold_hypo, args.threshold_hyper)
    ts_hypo, ts_hyper, ts_measured = report_time_spent([di[2] for di in sampled_datasets_bs], 'do(1,0)',
                                                       args.threshold_hypo, args.threshold_hyper)
    ts_hypo, ts_hyper, ts_measured = report_time_spent([di[3] for di in sampled_datasets_bs], 'do(1,1)',
                                                       args.threshold_hypo, args.threshold_hyper)
    print('Printing Operation statistics...')
    sampled_datasets_op = sampled_datasets[n_day_bs:]
    ts_hypo, ts_hyper, ts_measured = report_time_spent([di[0] for di in sampled_datasets_op], 'do(0,0)',
                                                       args.threshold_hypo, args.threshold_hyper)
    ts_hypo, ts_hyper, ts_measured = report_time_spent([di[1] for di in sampled_datasets_op], 'do(0,1)',
                                                       args.threshold_hypo, args.threshold_hyper)
    ts_hypo, ts_hyper, ts_measured = report_time_spent([di[2] for di in sampled_datasets_op], 'do(1,0)',
                                                       args.threshold_hypo, args.threshold_hyper)
    ts_hypo, ts_hyper, ts_measured = report_time_spent([di[3] for di in sampled_datasets_op], 'do(1,1)',
                                                       args.threshold_hypo, args.threshold_hyper)


def report_time_spent(dataset, ds_name, threshold_hypo, threshold_hyper):
    ts_hypo, ts_hyper, ts_measured = 0, 0, 0
    for di in dataset:
        y = di[1]
        ts_hypo += np.sum(y <= threshold_hypo)
        ts_hyper += np.sum(y >= threshold_hyper)
        ts_measured += len(y)
    print(f'[{ds_name}], ts[hypo]={ts_hypo} ({ts_hypo/ts_measured}%), '
          f'ts[hyper]={ts_hyper} ({ts_hyper/ts_measured}%), '
          f'ts[measured]={ts_measured}')
    return ts_hypo, ts_hyper, ts_measured


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampler_dir', type=str, default='models.sampler')
    parser.add_argument('--samples_dir', type=str, default='samples')
    parser.add_argument('--use_time_corrections', action='store_true')
    parser.add_argument('--tc_folder', type=str,)
    parser.add_argument('--seed', type=int, default=1)
    init_args = parser.parse_args()
    init_dict = vars(init_args)
    init_dict.update(constants.PARAMS_GENERAL)
    init_dict.update(constants.PARAMS_SAMPLING_RW)
    init_dict.update(constants.GPRPP_FAO)
    init_args = argparse.Namespace(**init_dict)
    init_args.samples_dir = os.path.join(init_args.samples_dir, f's{init_args.seed}')
    os.makedirs(init_args.samples_dir, exist_ok=True)
    np.random.seed(init_args.seed)
    tf.random.set_seed(init_args.seed)
    run_real_world_sampling_task(init_args)
