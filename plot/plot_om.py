import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
import numpy as np

from utils.data_utils import get_data_slice

# plt.rcParams['text.usetex'] = True


def plot_fs_pred(xnew, f_means, f_vars, f_labels, f_colors, ds, show_data=True, path=None, plot_var=True):
    fig, ax1 = plt.subplots(figsize=(15, 4))
    lines, labels = [], f_labels
    for fm, fv, col in zip(f_means, f_vars, f_colors):
        line_gp = plot_gp_pred(xnew, fm, fv, color=col, plot_var=plot_var)
        lines.append(line_gp)
    ax2 = ax1.twinx()
    if show_data:
        line1, bar2 = plot_joint_data(ds, axes=(ax1, ax2))
        lines = [line1, bar2] + lines
        labels = [r'$y(\tau)$', r'$\mathbf{a}$'] + labels
    ax1.legend(lines, labels, loc=2, fontsize=14, framealpha=1.0)
    plt.savefig(path)
    plt.close()


def plot_fs_pred_multiple(xnew, f_means, f_vars, ds_plots, period, path, args, plot_var=True, ds_type='train'):
    offset = 0
    ds_mse, ds_mae = {}, {}
    for i, ds_plot in enumerate(ds_plots):
        predict_shape = xnew[i].shape[0]
        f_means_i = [ff[offset:offset + predict_shape] for ff in f_means]
        f_vars_i = [ff[offset:offset + predict_shape] for ff in f_vars]
        file_name = f'f_{ds_type}_fit_p{period}_id{args.patient_ids[i]}_v{plot_var}.pdf'
        y_i, f_i = ds_plot[1], f_means_i[-1].numpy().flatten()
        mse_i = np.mean((y_i - f_i) ** 2)
        mae_i = np.mean(np.abs(y_i - f_i))
        ds_mse[args.patient_ids[i]] = [mse_i]
        ds_mae[args.patient_ids[i]] = [mae_i]
        plot_fs_pred(xnew[i], f_means_i, f_vars_i,
                     [r'$\mathbf{f_b}$', r'$\mathbf{f_a}$', r'$\mathbf{f}$'],
                     ['tab:orange', 'tab:green', 'tab:blue'], ds_plot,
                     show_data=True, plot_var=plot_var, path=os.path.join(path, file_name))
        offset += predict_shape
    pd.DataFrame.from_dict(ds_mse).to_csv(os.path.join(args.output_dir, f'mse_{ds_type}.csv'))
    pd.DataFrame.from_dict(ds_mae).to_csv(os.path.join(args.output_dir, f'mae_{ds_type}.csv'))
    print(f"MSE: {np.mean([v for v in ds_mse.values()])}")
    print(f"MAE: {np.mean([v for v in ds_mae.values()])}")


def plot_ft_pred(xnew, f_mean, f_var, path=None):
    plt.figure(figsize=(10, 5))
    plot_gp_pred(xnew, f_mean, f_var, label=r'$f_a(\tau, (t_0,m_0)=(0.0, m))$')
    plt.legend(fontsize=18, loc='upper right', framealpha=1.0)
    plt.xlabel(r'Time (hours), $\tau$', fontsize=28)
    plt.ylabel(r'Glucose (mmol/l), $f_a(\cdot)$', fontsize=28)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_ft_comparison(xnew, f_means, f_vars, mark, path=None):
    plt.figure(figsize=(10, 5))
    lines = []
    handles = [r'$f_a$ Baseline', r'$f_a$ Operation']
    fmax = np.max(np.concatenate(f_means))
    for f_mean, f_var, label, color in zip(f_means, f_vars,  handles,
                                           ['tab:blue', 'tab:red']):
        line_gp = plot_gp_pred(xnew, f_mean, f_var, color=color, label=label)
        lines.append(line_gp)
    plt.xlim(-0.2, np.max(xnew)+0.05)
    plt.ylim(-fmax/3, 6/5*fmax)
    ax1 = plt.gca()
    ax12 = ax1.twinx()
    ax1.set_xticks([0, 1, 2, 3])
    ax1.set_yticks(np.linspace(0.0, fmax, 5))
    ax1.set_xlabel(r'Time (hours), $\tau$', fontsize=24)
    ax1.set_ylabel(r'Glucose (mmol/l), $f_a(\cdot)$', fontsize=24)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.tick_params(labelsize=16)
    label2 = r'Treatment $a$'
    ax12.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    bar2 = ax12.bar([0.0], [mark], color='cyan', edgecolor='black', width=0.1, lw=3)
    ax12.set_ylim(0.0, mark/4*23)
    ax12.set_yticks(np.linspace(0.0, mark/4*5, 4))
    ax12.tick_params(labelsize=16)
    ax12.set_ylabel('Carb. intake '+'(log g)'+', $m$', fontsize=24)
    plt.legend([bar2]+lines, [label2]+handles, fontsize=18, loc='upper right', framealpha=1.0)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def compare_f_pred_multiple(fb_means, f_means, fb_vars, f_vars, ds_trains, period, path, args):
    n_patient = len(ds_trains[0])
    hours_day = 24.0
    time_offset = 2.0
    offset_bs, offset_op = 0, 0
    for i in range(n_patient):
        x1, y1, t1, m1 = ds_trains[0][i]
        x2, y2, t2, m2 = ds_trains[1][i]
        m1, m2 = np.exp(m1), np.exp(m2)
        x2 += hours_day + time_offset
        t2 += hours_day + time_offset
        mmax = np.max(np.concatenate([m1, m2]))
        predict_shape1, predict_shape2 = x1.shape[0], x2.shape[0]
        f_mean1, f_var1 = f_means[0][offset_bs:offset_bs + predict_shape1], f_vars[0][offset_bs:offset_bs + predict_shape1]
        f_mean2, f_var2 = f_means[1][offset_bs:offset_bs + predict_shape2], f_vars[1][offset_bs:offset_bs + predict_shape2]
        fb_mean1, fb_var1 = fb_means[0][offset_bs:offset_bs + predict_shape1], fb_vars[0][offset_bs:offset_bs + predict_shape1]
        fb_mean2, fb_var2 = fb_means[1][offset_bs:offset_bs + predict_shape2], fb_vars[1][offset_bs:offset_bs + predict_shape2]

        offset_bs += predict_shape1
        offset_op += predict_shape2

        fig = plt.figure(figsize=(15, 4))
        ax1 = plt.gca()
        ax1_xlim = (0.0, 2 * hours_day + time_offset)
        lines1, = ax1.plot(x1, y1, 'x', color='black', label=r"Glucose, $\mathbf{y}$", zorder=10)
        ax1.plot(x1, f_mean1, color='tab:blue', ls='--', label=r"$\mathbf{f}$", zorder=15)
        ax1.fill_between(
            x1,
            f_mean1[:, 0] - 1.96 * np.sqrt(f_var1[:, 0]),
            f_mean1[:, 0] + 1.96 * np.sqrt(f_var1[:, 0]),
            color='tab:blue',
            alpha=0.2,
            zorder=15,
        )
        ax1.plot(x1, fb_mean1, color='lightcoral', ls='--', label=r"$\mathbf{f_b}$")
        ax1.fill_between(
            x1,
            fb_mean1[:, 0] - 1.96 * np.sqrt(fb_var1[:, 0]),
            fb_mean1[:, 0] + 1.96 * np.sqrt(fb_var1[:, 0]),
            color='lightcoral',
            alpha=0.2,
        )
        ax1.plot(x2, y2, 'x', color='black', zorder=10)
        ax1.plot(x2, f_mean2, color='tab:blue', ls='--', zorder=15)
        ax1.fill_between(
            x2,
            f_mean2[:, 0] - 1.96 * np.sqrt(f_var2[:, 0]),
            f_mean2[:, 0] + 1.96 * np.sqrt(f_var2[:, 0]),
            color='tab:blue',
            alpha=0.2,
            zorder=15
        )
        ax1.plot(x2, fb_mean2, color='lightcoral', ls='--', )
        ax1.fill_between(
            x2,
            fb_mean2[:, 0] - 1.96 * np.sqrt(fb_var2[:, 0]),
            fb_mean2[:, 0] + 1.96 * np.sqrt(fb_var2[:, 0]),
            color='lightcoral',
            alpha=0.2,
        )
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax1.set_xlim(*ax1_xlim)
        ax1.tick_params(labelsize=16)
        ax1.set_ylabel(r'Glucose (mmol/l), $y$', fontsize=26)
        ax1.set_xlabel(r'Time, $\tau$', fontsize=26)

        ax1.axvspan(hours_day, hours_day + time_offset, color="grey", alpha=0.1)
        ax1_ylim = ax1.get_ylim()
        ax1_ylim = (ax1_ylim[0] - 1, ax1_ylim[1])
        ax1.set_ylim(*ax1_ylim)
        ax1.vlines([hours_day, hours_day + time_offset], *ax1_ylim, linestyle="--", color="grey")
        ax1.vlines([hours_day + time_offset / 2], *ax1_ylim, linestyle="--", color="black", lw=2)
        ax1.annotate("", xy=(25.0, ax1_ylim[1] - 0.1), xytext=(25.0, ax1_ylim[1] + 0.6),
                     arrowprops=dict(arrowstyle="simple", connectionstyle="arc3"), annotation_clip=False)
        ax1.text(3.0, ax1_ylim[1],
                 r"$\underbrace{\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad}_{}$",
                 fontsize=22, rotation=180)
        ax1.text(7.6, ax1_ylim[1] + 0.5, r"\textsc{Pre-surgery}", fontsize=22)
        ax1.text(30.0, ax1_ylim[1],
                 r"$\underbrace{\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad}_{}$",
                 fontsize=22, rotation=180)
        ax1.text(35.3, ax1_ylim[1] + 0.5, r"\textsc{Post-surgery}", fontsize=22)
        text = ax1.text(25.0, ax1_ylim[1] + 0.6, r"\textsc{Intervention}",
                        color="black", fontsize=22,
                        horizontalalignment="center", verticalalignment="center",
                        bbox=dict(boxstyle="round", fc="white", ec="black", pad=0.2))

        ax12 = ax1.twinx()
        bar2 = ax12.bar(t1, m1, color='cyan', edgecolor='black', width=0.4, lw=2,
                        label=r'Meals, $\mathbf{m}$')
        bar2 = ax12.bar(t2, m2, color='cyan', edgecolor='black', width=0.4, lw=2, )
        ax12.set_ylabel('Carb. intake (g), $m$', fontsize=26)
        ax12.set_ylim(0.0, mmax * (ax1_ylim[1] - ax1_ylim[0]))
        ax12.set_yticks([10.0, 30.0, 50.0])
        ax12.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        # ax12.set_yticklabels(['0.0', '2.0', '4.0'])
        ax12.tick_params(labelsize=16)
        h, l = [(a + b) for a, b in zip(ax1.get_legend_handles_labels(), ax12.get_legend_handles_labels())]
        order_idx = [0, 3, 1, 2]
        lgd = ax1.legend([h[i] for i in order_idx], [l[i] for i in order_idx], fontsize=18, loc='upper right',
                         framealpha=1.0, ncol=4, bbox_to_anchor=(0.78, -0.2))
        file_name = f'f_train_fit_p{period}_id{args.patient_ids[i]}.pdf'
        fig.savefig(os.path.join(path, file_name), bbox_extra_artists=(lgd, text,), bbox_inches='tight')
        plt.close()


def plot_gp_pred(xnew, f_mean, f_var, color='tab:blue', label='f Pred', plot_var=True):
    line_gp, = plt.plot(xnew, f_mean, color, ls='--', lw=2, label=label, zorder=2)
    if plot_var:
        plt.fill_between(
            xnew[:, 0],
            f_mean[:, 0] - 1.96 * np.sqrt(f_var[:, 0]),
            f_mean[:, 0] + 1.96 * np.sqrt(f_var[:, 0]),
            color=color,
            alpha=0.2,
        )
    return line_gp


def plot_joint_data(ds, axes=None):
    if axes is None:
        fig, ax1 = plt.subplots(figsize=(15, 4))
        ax2 = ax1.twinx()

    x, y, t, m = ds[0], ds[1], ds[2], ds[3]

    day_min, day_max = x.min() // 24, x.max() // 24
    meal_bar_width = (day_max - day_min + 1) * (1 / 6)
    ax1, ax2 = axes
    #
    line1, = ax1.plot(x, y, 'kx', ms=5, alpha=0.5, label='Glucose, $y(t)$')
    ylim1 = ax1.get_ylim()
    ax1.set_ylim(ylim1[0] - 1.0, ylim1[1])
    bar2 = ax2.bar(t, m, color=(0.1, 0.1, 0.1, 0.1), edgecolor='red', width=meal_bar_width,
                   label=r'Meal, $\mathbf{a}$')
    # Widen ylim2, so that max meal is under min glucose
    ylim2 = ax2.get_ylim()
    ylim_max2_wide = ylim2[1] * (1.0+ylim1[1]-ylim1[0])
    if np.isfinite(ylim2[0]):
        ax2.set_ylim(ylim2[0], ylim_max2_wide)
    #
    ax1.vlines([24 * i for i in range(int(day_min), int(day_max) + 1)], 0.0, ylim1[1], colors='grey', alpha=0.5)
    return line1, bar2


def compare_f_preds(Xnew, f_means, labels, ylabel, ylim=None, path=None):
    plt.figure(figsize=(6, 3))
    for i, (f_mean, label) in enumerate(zip(f_means, labels)):
        plt.plot(Xnew, f_mean, '--', lw=3, label=label)
    plt.legend(fontsize=16)
    plt.xlabel(r'Time, $\tau$', fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()



