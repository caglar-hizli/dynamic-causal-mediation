import os
import numpy as np
import tensorflow as tf

from plot.plot_om import plot_ft_pred, plot_ft_comparison, plot_fs_pred_multiple, plot_fs_pred, \
    compare_f_pred_multiple, compare_f_preds


def predict_ft(model, args, plot_path=None):
    xnew = np.linspace(0.0, args.T_treatment, 100).reshape(-1, 1)
    tnew = np.array([-0.001])
    ft_mean, ft_var = model.predict_ft(xnew, tnew)
    plot_ft_pred(xnew, ft_mean, ft_var, path=plot_path)
    return ft_mean, ft_var


def compare_ft(models, args, plot_path=None):
    xnew = np.linspace(0.0, args.T_treatment, 100).reshape(-1, 1)
    tnew = np.array([-0.001])
    ft_means, ft_vars = [], []
    for model in models:
        ft_mean, ft_var = model.predict_ft(xnew, tnew)
        ft_means.append(ft_mean)
        ft_vars.append(ft_var)
    plot_ft_comparison(xnew, ft_means, ft_vars, mark=1.0, path=plot_path)
    return ft_means, ft_vars


def predict_ft_marked(model, args):
    xnew = np.linspace(0.0, args.T_treatment, 100).reshape(-1, 1)
    for m in [1.0, 3.0, 5.0, 10.0]:
        tnew = np.array([-0.001, m]).reshape(-1, 2)
        ft_mean, ft_var = model.predict_ft_compiled(xnew, tnew)
        plot_ft_pred(xnew, ft_mean, ft_var, path=os.path.join(args.output_dir, f'predict_ft_m{m:.1f}.pdf'))
    return ft_mean, ft_var


def predict_ft_marked_hier(model, args):
    xnew = np.linspace(0.0, args.T_treatment, 100).reshape(-1, 1)
    for porder, pidx in enumerate(args.patient_ids):
        pidx_tf = tf.ones(1, dtype=tf.int32) * pidx
        for m in [1.0, 3.0, 5.0, 10.0]:
            tnew = np.array([-0.001, m]).reshape(-1, 2)
            ft_mean, ft_var = model.predict_ft_single_for_patient_compiled(xnew, tnew, pidx_tf[0])
            ft_var = np.diag(ft_var.numpy()[0]).reshape(-1, 1)
            plot_ft_pred(xnew, ft_mean, ft_var,
                         path=os.path.join(args.output_figures_dir, f'predict_ft_p{pidx}_m{m:.1f}.pdf'))


def compare_fb(models, path, args):
    n_test = 100
    n_patients = len(args.patient_ids)
    x_flat = np.linspace(0.0, 24.0, n_test).reshape(-1, 1)
    xnew = [x_flat for _ in range(n_patients)]
    fb_means, fb_vars = [], []
    for model in models:
        fb_mean, fb_var = model.predict_baseline_compiled(xnew)
        fb_mean = np.stack([fb_mean[i*n_test:(i+1)*n_test] for i in range(n_patients)]).sum(0) / n_patients
        fb_var = np.stack([fb_var[i*n_test:(i+1)*n_test] for i in range(n_patients)]).sum(0) / (n_patients**2)
        fb_means.append(fb_mean)
        fb_vars.append(fb_var)
    plot_fs_pred(x_flat, fb_means, fb_vars,
                 [r'Baseline $\mathbf{f_b}$', r'Operation $\mathbf{f_b}$'], ['tab:blue', 'tab:red'],
                 ds=None, show_data=False, path=os.path.join(path, f'compare_fb.pdf'))


def compare_fb_period(model, period, path, args):
    N_test = 100
    n_patients = len(args.patient_ids)
    xnew = [np.linspace(0.0, 24.0, N_test).astype(np.float64).reshape(-1, 1) for _ in range(n_patients)]
    fb_mean, fb_var = model.predict_baseline_compiled(xnew)
    fb_means = [fb_mean[i*N_test:(i+1)*N_test] for i in range(n_patients)]
    compare_f_preds(xnew[0], fb_means, ylim=(0.0, 6.0),
                    ylabel=r'Glucose Baseline, $f_b(\tau)$',
                    labels=[r'$f_{b,gr_' + f'{i+1}' + r'}$' for i in range(n_patients)],
                    path=os.path.join(path, f'compare_fb_{period}.pdf'))


def compare_ft_marked_hier_period(model, period, path, args):
    xnew = np.linspace(0.0, args.T_treatment, 100).reshape(-1, 1)
    n_patients = len(args.patient_ids)
    m = 3.0
    ft_means = []
    for pidx in args.patient_ids:
        tnew = np.array([-0.001, m]).reshape(-1, 2)
        pidx_tf = tf.ones(1, dtype=tf.int32) * pidx
        ft_mean, _ = model.predict_ft_single_for_patient_compiled(xnew, tnew, pidx_tf[0])
        print(pidx, np.mean(ft_mean))
        ft_means.append(ft_mean)
    compare_f_preds(xnew, ft_means, ylim=(0.0, 6.0),
                    ylabel=r'Mediator Response, $f_m(\tau)$',
                    labels=[r'$f_{m,gr_' + f'{i + 1}' + r'}$' for i in range(n_patients)],
                    path=os.path.join(path, f'compare_ft_m{int(m)}_{period}.pdf'))


def compare_ft_marked_hier(models, path, args):
    xnew = np.linspace(0.0, args.T_treatment, 100).reshape(-1, 1)
    for porder, pidx in enumerate(args.patient_ids):
        pidx_tf = tf.ones(1, dtype=tf.int32) * pidx
        for m in [1.0, 3.0, 5.0, 10.0]:
            tnew = np.array([-0.001, m]).reshape(-1, 2)
            ft_means, ft_vars = [], []
            for model in models:
                ft_mean, ft_var = model.predict_ft_single_for_patient_compiled(xnew, tnew, pidx_tf[0])
                ft_var = np.diag(ft_var.numpy()[0]).reshape(-1, 1)
                ft_means.append(ft_mean)
                ft_vars.append(ft_var)
            plot_ft_comparison(xnew, ft_means, ft_vars, mark=m,
                               path=os.path.join(path, f'compare_ft_p{pidx}_m{m:.1f}.pdf'))


def compare_ft_marked(models, path, args):
    xnew = np.linspace(0.0, args.T_treatment, 100).reshape(-1, 1)
    for m in [1.0, 3.0, 5.0, 10.0]:
        tnew = np.array([-0.001, m]).reshape(-1, 2)
        ft_means, ft_vars = [], []
        for model in models:
            ft_mean, ft_var = model.predict_ft_compiled(xnew, tnew)
            ft_means.append(ft_mean)
            ft_vars.append(ft_var)
        plot_ft_comparison(xnew, ft_means, ft_vars, mark=m, path=os.path.join(path, f'compare_ft_m{m:.1f}.pdf'))
    return ft_means, ft_vars


def predict_f_train_fit(model, ds_plot, plot_path=None):
    x, y, t, m = ds_plot
    xnew = x.astype(np.float64).reshape(-1, 1)
    tnew = t.astype(np.float64)
    ft_mean, ft_var = model.predict_ft_w_tnew(xnew, tnew)
    fb_mean, fb_var = model.predict_baseline(xnew)
    f_mean, f_var = model.predict_f_train(xnew)
    plot_fs_pred(xnew, [fb_mean, ft_mean, f_mean], [fb_var, ft_var, f_var],
                 [r'$\mathbf{f_b}$', r'$\mathbf{f_a}$', r'$\mathbf{f}$'],
                 ['tab:orange', 'tab:green', 'tab:blue'],
                 ds_plot, show_data=True, path=plot_path)
    # plot_f_pred(xnew, ft_mean+fb_mean, ft_var+fb_var, ds_plot, show_data=True, path=plot_path)
    return ft_mean, ft_var


def predict_f_train_fit_shared(model, ds_plots, period, path, args):
    xnew = [ds[0].astype(np.float64).reshape(-1, 1) for ds in ds_plots]
    anew = [ds[2].astype(np.float64) for ds in ds_plots]
    ft_mean, ft_var = model.predict_ft_w_tnew(xnew, anew)
    fb_mean, fb_var = model.predict_baseline(xnew)
    f_mean, f_var = ft_mean+fb_mean, ft_var+fb_var
    plot_fs_pred_multiple(xnew, [fb_mean, ft_mean, f_mean], [fb_var, ft_var, f_var], ds_plots, period, path, args)
    return ft_mean, ft_var


def predict_f_train_fit_shared_marked(model, ds_plots, period, path, args):
    xnew = [ds[0].astype(np.float64).reshape(-1, 1) for ds in ds_plots]
    anew = [np.hstack([ds[2].astype(np.float64).reshape(-1, 1),
                       ds[3].astype(np.float64).reshape(-1, 1)]) for ds in ds_plots]
    ft_mean, ft_var = model.predict_ft_w_tnew_compiled(xnew, anew)
    ft_var = np.diag(ft_var.numpy()[0]).reshape(-1, 1)
    fb_mean, fb_var = model.predict_baseline_compiled(xnew)
    f_mean, f_var = ft_mean + fb_mean, ft_var + fb_var
    plot_fs_pred_multiple(xnew, [fb_mean, ft_mean, f_mean], [fb_var, ft_var, f_var], ds_plots, period, path, args)
    return ft_mean, ft_var


def get_f_train_fit_shared_marked_hier(model, ds_plots):
    xnew = [ds[0].astype(np.float64).reshape(-1, 1) for ds in ds_plots]
    anew = [np.hstack([ds[2].astype(np.float64).reshape(-1, 1),
                       ds[3].astype(np.float64).reshape(-1, 1)]) for ds in ds_plots]
    t_lengths = [ti.shape[0] for ti in anew]
    Np = len(t_lengths)
    patient_order_arr = np.arange(Np, dtype=np.int32)
    tnew_patient_idx = np.repeat(patient_order_arr, t_lengths)
    ft_mean, ft_var = model.predict_ft_w_tnew_compiled(xnew, anew, tnew_patient_idx)
    ft_var = np.diag(ft_var.numpy()[0]).reshape(-1, 1)
    fb_mean, fb_var = model.predict_baseline_compiled(xnew)
    f_mean, f_var = ft_mean + fb_mean, ft_var + fb_var
    return xnew, fb_mean, fb_var, ft_mean, ft_var, f_mean, f_var


def predict_f_train_fit_shared_marked_hier(model, ds_plots, period, path, args, ds_type='train'):
    xnew, fb_mean, fb_var, ft_mean, ft_var, f_mean, f_var = get_f_train_fit_shared_marked_hier(model, ds_plots)
    plot_fs_pred_multiple(xnew, [fb_mean, ft_mean, f_mean], [fb_var, ft_var, f_var], ds_plots, period, path, args,
                          ds_type=ds_type)
    return f_mean, f_var


def compare_train_fit_shared_marked_hier(models, ds_trains, period, path, args):
    xs, fb_means, fb_vars, f_means, f_vars = [], [], [], [], []
    for model, ds_train in zip(models, ds_trains):
        xnew, fb_mean, fb_var, _, _, f_mean, f_var = get_f_train_fit_shared_marked_hier(model, ds_train)
        xs.append(xnew)
        fb_means.append(fb_mean)
        fb_vars.append(fb_var)
        f_means.append(f_mean)
        f_vars.append(f_var)

    # compare_f_pred_multiple(fb_means, f_means, fb_vars, f_vars, ds_trains, period, path, args)
