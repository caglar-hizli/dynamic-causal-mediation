import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from simulate.semi_synth.sample_multiple import get_lambda_x, get_f_mean


# plt.rcParams['text.usetex'] = True


def plot_path_specific_trajectories(models_cf, model_types, exp_ids, multiple_ds, algorithm_log,
                                    pidx, figures_path, time_domain, plot_dict, args):
    n_exp = len(exp_ids)
    n_model_tot = len(models_cf)
    n_model_exp = int(n_model_tot // n_exp)
    plot_ids_y00 = [i*n_model_exp for i in range(n_exp)]
    plot_str = 'y00'
    plot_joint_trajectories(models_cf, model_types, exp_ids, plot_dict, multiple_ds, plot_ids_y00, plot_str,
                            algorithm_log, pidx, time_domain, figures_path, args)
    plot_ids_y11 = [i * n_model_exp + (n_model_exp-1) for i in range(n_exp)]
    plot_str = 'y11'
    plot_joint_trajectories(models_cf, model_types, exp_ids, plot_dict, multiple_ds, plot_ids_y11, plot_str,
                            algorithm_log, pidx, time_domain, figures_path, args)
    plot_ids_y01 = [i * n_model_exp + 1 for i in range(n_exp)]
    plot_str = 'y01'
    plot_joint_trajectories(models_cf, model_types, exp_ids, plot_dict, multiple_ds, plot_ids_y01, plot_str,
                            algorithm_log, pidx, time_domain, figures_path, args)
    plot_ids_y10 = [i * n_model_exp + 2 for i in range(n_exp)]
    plot_str = 'y10'
    plot_joint_trajectories(models_cf, model_types, exp_ids, plot_dict, multiple_ds, plot_ids_y10, plot_str,
                            algorithm_log, pidx, time_domain, figures_path, args)


def plot_joint_trajectories(models_cf, model_types_cf, exp_ids, plot_dict, patient_datasets, plot_ids, plot_str,
                            algorithm_log, pidx, period, out_folder, args):
    models = [models_cf[i] for i in plot_ids]
    model_types = [model_types_cf[i] for i in plot_ids]
    accept_noise = np.array([l[3][1] for l in algorithm_log if l[2][1]])
    candidates_lambda_ub = np.array([l[1] for l in algorithm_log if l[2][1]])
    lambda_ub = [l[1] for l in algorithm_log]
    baseline_time = np.array([period[0]])
    accept_noise_scaled = accept_noise * candidates_lambda_ub
    x_ub = [l[0][1] for l in algorithm_log]
    _ = plt.subplots(2, 1, sharex=True, figsize=(15, 6), gridspec_kw={'height_ratios': [1.5, 1]})
    #
    plt.subplot(2, 1, 1)
    ts = [patient_datasets[i][2] for i in plot_ids]
    ms = [patient_datasets[i][3] for i in plot_ids]
    ys = [patient_datasets[i][1] for i in plot_ids]
    x = patient_datasets[0][0]
    mall = np.concatenate(ms)
    mmax = np.max(mall) if len(mall) > 0 else 1.0
    ymin = np.min(np.concatenate(ys))
    fs_legend = 13
    fs_label = 16
    ii = 0
    colors = mcolors.TABLEAU_COLORS
    color_names = ['tab:blue', 'tab:pink', 'tab:purple', 'tab:brown', 'tab:orange']
    for i, (model, to, mo, yo, model_type, exp_id) \
            in enumerate(zip(models, ts, ms, ys, model_types, exp_ids)):
        if exp_id == 'oracle':
            alpha = 0.7
        elif exp_id == 'interventional':
            alpha = 1.0
        else:
            alpha = 0.25
        lw = 4 if exp_id in ['oracle', 'interventional'] else 2
        bottom = ymin-0.25-0.25 * (ii+1)
        ls = '-' if exp_id in ['oracle'] else '--'
        if exp_id in plot_dict:
            cn = color_names[ii]
            plt.plot(x, yo, ls+'o', color=colors[cn], alpha=alpha, lw=lw,
                     label=r'$\mathbf{Y}_{' + plot_dict[exp_id] + '}$')
            plt.bar(to, (mo / mmax) / 4, bottom=bottom, color=colors[cn], edgecolor='black',
                    width=0.2, lw=2, alpha=alpha,
                    label=r'$\mathbf{m}_{' + plot_dict[exp_id] + '}$')
            ii += 1
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        labelbottom=False)
    plt.ylabel(r'Glucose (mmol/l), $Y$', fontsize=fs_label)
    plt.legend(fontsize=fs_legend, loc='upper left', framealpha=1.0, ncol=2)

    y_norms = [patient_datasets[i][4] for i in plot_ids]
    #
    plt.subplot(2, 1, 2)
    plt.step([period[0]]+x_ub, [lambda_ub[0]]+lambda_ub, '-',
             color='grey', lw=2, label=r'$\lambda_{ub}$', alpha=0.5)
    candidates = [l[2][0] for l in algorithm_log if l[2][1]]
    plt.plot(candidates, accept_noise_scaled, "rx", markersize=13, label='Accept')
    plt.vlines(candidates, np.zeros_like(candidates_lambda_ub), accept_noise_scaled,
               lw=1, colors='red', alpha=0.75)

    ii = 0
    for i, (model, action_time, y_norm, model_type, exp_id) in \
            enumerate(zip(models, ts, y_norms, model_types, exp_ids)):
        if exp_id == 'oracle':
            alpha = 0.7
        elif exp_id == 'interventional':
            alpha = 1.0
        else:
            alpha = 0.25
        if exp_id in plot_dict:
            cn = color_names[ii]
            ls = '-' if exp_id in ['oracle'] else '--'
            time_intensity = model[0][0]
            outcome_norm = np.stack([x, y_norm]).T
            x_all = np.sort(np.concatenate([action_time, x, candidates]))
            lambdaXa_o = get_lambda_x(time_intensity, model_type[0], x_all,
                                      baseline_time, action_time, outcome_norm, args)
            plt.plot(x_all, lambdaXa_o, ls, color=colors[cn], lw=2, alpha=alpha,
                     label=r'$\lambda_{' + plot_dict[exp_id] + '}$')
            ii += 1

    plt.legend(fontsize=fs_legend-2, loc='upper left', framealpha=1.0)
    plt.xticks(np.arange(period[0], period[1]+1))
    plt.ylabel(r'Intensity, $\lambda(\tau)$', fontsize=fs_label)
    plt.xlabel(r'Time (hours), $\tau$', fontsize=fs_label)
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, f'compare_multiple_sampling_{pidx}_{plot_str}.pdf'))
    plt.close()


def plot_outcome_benchmarks(outcome_model, om_type, patient_dataset, period, pidx, out_folder, args, ds_type='train',
                            x_grid=None):
    x, y, t, m = patient_dataset[0], patient_dataset[1], patient_dataset[2], patient_dataset[3]
    action = np.stack([t, m]).T
    plt.figure(figsize=(12, 4))
    f_mean = get_f_mean(outcome_model, om_type=om_type, x=x_grid if x_grid is not None else x, action=action,
                        patient_idx=pidx, fb_mean=None, period=period, args=args)
    plt.plot(x, y, 'o', color='tab:blue', alpha=0.75, lw=2, label=r'Y_train')
    plt.plot(x_grid if x_grid is not None else x, f_mean, '-', color='tab:orange', alpha=0.75, lw=2, label=r'Y_pred')
    mmax = np.max(m) if len(m) > 0 else 1.0
    plt.bar(t, (m / mmax), bottom=2.0, color='tab:blue', edgecolor='black', width=0.2, lw=2, alpha=0.5)
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        labelbottom=False)
    plt.ylabel(r'Glucose (mmol/l), $Y$', fontsize=16)
    plt.legend(fontsize=13, loc='lower right', framealpha=1.0)

    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, f'outcome_{ds_type}_fit_{om_type}_{pidx}.pdf'))
    plt.close()
    return f_mean, y
