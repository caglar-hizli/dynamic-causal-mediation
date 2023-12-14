import os

import tensorflow as tf
import gpflow as gpf

from utils.om_utils import parse_om_init_args, save_model_shared_marked_hier, prepare_om_input_marked
from train.outcome.predict import predict_f_train_fit_shared_marked_hier, compare_fb_period, \
    compare_ft_marked_hier_period, compare_train_fit_shared_marked_hier, predict_ft_marked_hier
from utils.om_utils import get_baseline_kernel, get_treatment_base_kernel, train_model
from utils.om_utils import get_patient_datasets
from models.outcome.piecewise_se.model_shared_marked_hierarchical import TRSharedMarkedHier
from utils.data_utils import get_updated_meta_df, get_joint_df


def train_ft(args):
    os.makedirs(args.output_dir, exist_ok=True)
    df_meta = get_updated_meta_df()
    models, ds_trains = [], []
    periods, data_slices = ['Baseline', 'Operation'], [1, 2]
    for period, data_slice in zip(periods, data_slices):
        df_joint = get_joint_df(period)
        ds_train, ds_test, ds_all = get_patient_datasets(df_meta, df_joint, period, data_slice, args)
        ds_trains.append(ds_train)
        model_folder = os.path.join(args.output_dir, period, 'outcome.p' + ','.join(str(i) for i in args.patient_ids))
        os.makedirs(model_folder, exist_ok=True)
        model_path = os.path.join(model_folder, 'outcome_model')
        if not os.path.exists(model_path):
            model = run_model(ds_train, args)
            outcome_model_path = os.path.join(model_folder, 'outcome_model')
            save_model_shared_marked_hier(model, outcome_model_path)
        model = tf.saved_model.load(model_path)
        models.append(model)
        figure_path = os.path.join(model_folder, 'figures')
        os.makedirs(figure_path, exist_ok=True)
        predict_f_train_fit_shared_marked_hier(model, ds_all, period=period, path=figure_path, args=args)

    for model in models:
        print(f'Log marginal ll: {model.log_marginal_likelihood_compiled().numpy().item():.3f}')
    compare_figures_dir = os.path.join(args.output_dir, args.period,
                                       'outcome.p' + ','.join(str(i) for i in args.patient_ids), 'figures')
    os.makedirs(compare_figures_dir, exist_ok=True)
    for model, period in zip(models, periods):
        compare_ft_marked_hier_period(model, period, compare_figures_dir, args)
        compare_fb_period(model, period, compare_figures_dir, args)
    compare_train_fit_shared_marked_hier(models, ds_trains, period='Compare', path=compare_figures_dir, args=args)


def train_ft_glucose_period(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    outcome_model_path = os.path.join(args.output_dir, 'outcome_model')
    df_meta = get_updated_meta_df()
    df_joint = get_joint_df(args.period)
    ds_trains, ds_tests, ds_alls = get_patient_datasets(df_meta, df_joint, args.period, args.data_slice, args)
    if not os.path.exists(outcome_model_path):
        model = run_model(ds_trains, args)
        print(f'Log marginal ll: {model.log_marginal_likelihood().numpy().item():.3f}')
        save_model_shared_marked_hier(model, outcome_model_path)
    model = tf.saved_model.load(outcome_model_path)
    predict_ft_marked_hier(model, args)
    predict_f_train_fit_shared_marked_hier(model, ds_alls, args.period, path=args.output_figures_dir, args=args,
                                           ds_type='all')


def run_model(ds_trains, args):
    model = build_om(ds_trains, args)
    model = train_model(model, args)
    return model


def build_om(ds_trains, args):
    x, y, a = prepare_om_input_marked(ds_trains)
    model = TRSharedMarkedHier(data=(x, y), t=a, T=args.T_treatment,
                               patient_ids=args.patient_ids,
                               baseline_kernels=[get_baseline_kernel() for _ in range(len(ds_trains))],
                               treatment_base_kernel=get_treatment_base_kernel(),
                               use_bias=args.use_bias,
                               mean_functions=[gpf.mean_functions.Zero() for _ in range(len(ds_trains))],
                               noise_variance=1.0,
                               train_noise=True, )
    return model


if __name__ == "__main__":
    init_args = parse_om_init_args()
    train_ft(init_args)
