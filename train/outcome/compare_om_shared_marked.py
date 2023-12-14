import os

import tensorflow as tf
import gpflow as gpf

from utils.om_utils import parse_om_init_args, save_model_shared_marked, prepare_om_input_marked
from train.outcome.predict import compare_ft_marked, compare_fb
from utils.om_utils import get_patient_datasets
from train.outcome.predict import predict_ft_marked, predict_f_train_fit_shared_marked
from utils.om_utils import get_baseline_kernel, get_treatment_base_kernel, train_model, \
    get_baseline_kernel_periodic
from models.outcome.piecewise_se.model_shared_marked import TRSharedMarked
from utils.data_utils import get_updated_meta_df, get_joint_df


def train_ft(args):
    os.makedirs(args.output_dir, exist_ok=True)
    df_meta = get_updated_meta_df()
    models = []
    for period, data_slice in zip(['Baseline', 'Operation'], [1, 2]):
        df_joint = get_joint_df(period)
        ds_train, ds_test, ds_all = get_patient_datasets(df_meta, df_joint, period, data_slice, args)
        model_folder = os.path.join(args.output_dir, period, 'outcome.p' + ','.join(str(i) for i in args.patient_ids))
        os.makedirs(model_folder, exist_ok=True)
        model_path = os.path.join(model_folder, 'outcome_model')
        if not os.path.exists(model_path):
            model = run_model(ds_all, args)
            save_model_shared_marked(model, model_folder)
        model = tf.saved_model.load(model_path)
        models.append(model)
        figure_path = os.path.join(model_folder, 'figures')
        os.makedirs(figure_path, exist_ok=True)
        predict_f_train_fit_shared_marked(model, ds_all, period=period, path=figure_path, args=args)

    for model in models:
        print(f'Log marginal ll: {model.log_marginal_likelihood_compiled().numpy().item():.3f}')
    compare_figures_dir = os.path.join(args.output_dir, args.period, 'figures')
    os.makedirs(compare_figures_dir, exist_ok=True)
    _, _ = compare_ft_marked(models, compare_figures_dir, args)
    compare_fb(models, compare_figures_dir, args)


def train_ft_glucose_period(args):
    os.makedirs(args.output_dir, exist_ok=True)
    #
    df_meta = get_updated_meta_df()
    df_joint = get_joint_df(args.period)
    ds_trains, ds_tests, ds_alls = get_patient_datasets(df_meta, df_joint, args.period, args.data_slice, args)
    #
    model = run_model(ds_trains, args)
    _, _ = predict_ft_marked(model, args)
    predict_f_train_fit_shared_marked(model, ds_alls, args.period, path=args.output_figures_dir, args=args)
    save_model_shared_marked(model, args.output_dir)
    if args.save_sampler:
        save_model_shared_marked(model, args.sampler_dir)


def run_model(ds_trains, args):
    model = build_outcome_model_glucose(ds_trains, args)
    model = train_model(model, args)
    return model


def build_outcome_model_glucose(ds_trains, args):
    x, y, a = prepare_om_input_marked(ds_trains)
    if args.share_baseline:
        baseline_kernel = get_baseline_kernel_periodic()
        baseline_kernels = [baseline_kernel+gpf.kernels.Constant() for _ in range(len(ds_trains))]
    else:
        baseline_kernels = [get_baseline_kernel() for _ in range(len(ds_trains))]
    model = TRSharedMarked(data=(x, y), t=a, T=args.T_treatment,
                           baseline_kernels=baseline_kernels,
                           treatment_base_kernel=get_treatment_base_kernel(),
                           mean_functions=[gpf.mean_functions.Zero() for _ in range(len(ds_trains))],
                           use_bias=args.use_bias)
    return model


if __name__ == "__main__":
    init_args = parse_om_init_args()
    train_ft(init_args)
