import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from utils.om_utils import get_patient_datasets
from utils.data_utils import get_updated_meta_df, get_joint_df

# plt.rcParams['text.usetex'] = True


def plot_samples(pidx, sample_patient_ids, args):
    args.remove_night_time = False
    datasets = []
    n_synth_patients = len(sample_patient_ids[0])
    for per, pids in zip(['Baseline', 'Operation'], sample_patient_ids):
        sample_path = os.path.join('triton.output.samples.v20/np50.ntr1.pa12.po12,25,28.no40.s1',
                                   per, 'patients.npz')
        dataset_dict = np.load(sample_path, allow_pickle=True)
        ds = dataset_dict['dataset']
        datasets.append([ds[i] for i in pids])
    fig, axes = plt.subplots(n_synth_patients, 1, figsize=(15, 9),
                             sharex=True, gridspec_kw={'height_ratios': [1]*n_synth_patients})
    for n in range(n_synth_patients):
        ds1, ds2 = datasets[0][n], datasets[1][n]
        plt.subplot(n_synth_patients, 1, n+1)
        plot_joint_data(ds1, ds2, annotate=False)
    plt.tight_layout()
    # patient_str = ','.join([str(s) for s in sample_patient_ids])
    plt.savefig(os.path.join(args.output_dir, f'appendix_fig6_synth_data_p{pidx}.pdf'))
    plt.close()


def plot_joint_data(ds1, ds2, annotate=True):
    hours_day, time_offset = 24.0, 2.0
    x1, y1, t1, m1 = ds1[0], ds1[1], ds1[2], ds1[3]
    x2, y2, t2, m2 = ds2[0], ds2[1], ds2[2], ds2[3]
    m1, m2 = np.exp(m1), np.exp(m2)
    x2 += hours_day + time_offset
    t2 += hours_day + time_offset
    mmax = np.max(np.concatenate([m1, m2]))

    ax1 = plt.gca()
    ax1_xlim = (0.0, 2 * hours_day + time_offset)
    lines1, = ax1.plot(x1, y1, 'x', color='black', label=r"Glucose, $\mathbf{y}$", zorder=10)
    ax1.plot(x2, y2, 'x', color='black', zorder=10)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.set_xlim(*ax1_xlim)
    ax1.tick_params(labelsize=16)
    ax1.set_ylabel(r'Glucose (mmol/l), $y$' if annotate else 'Glucose, $y$', fontsize=26)
    ax1.set_xlabel(r'Time, $\tau$', fontsize=26)

    ax1.axvspan(hours_day, hours_day + time_offset, color="grey", alpha=0.1)
    ax1_ylim = ax1.get_ylim()
    ax1_ylim = (ax1_ylim[0] - 1.0, ax1_ylim[1])
    print(ax1_ylim)
    ax1.set_ylim(*ax1_ylim)
    ax1.vlines([hours_day, hours_day + time_offset], *ax1_ylim, linestyle="--", color="grey")
    ax1.vlines([hours_day + time_offset / 2], *ax1_ylim, linestyle="--", color="black", lw=2)
    if annotate:
        ax1.annotate("", xy=(25.0, ax1_ylim[1] - 0.1), xytext=(25.0, ax1_ylim[1] + 0.6),
                     arrowprops=dict(arrowstyle="simple", connectionstyle="arc3"), annotation_clip=False)
        ax1.text(3.0, ax1_ylim[1]-0.1,
                 r"$\underbrace{\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad}_{}$",
                 fontsize=22, rotation=180)
        ax1.text(7.6, ax1_ylim[1] + 0.5, r"\textsc{Pre-surgery}", fontsize=22)
        ax1.text(30.0, ax1_ylim[1]-0.1,
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
    ax12.set_ylabel('Carb. intake (g), $m$' if annotate else 'Carb., $m$', fontsize=26)
    ylim_diff = ax1_ylim[1] - ax1_ylim[0] - 2.0 if annotate else (ax1_ylim[1] - ax1_ylim[0]) * 1.5
    ax12.set_ylim(0.0, mmax * ylim_diff)
    ax12.set_yticks([25.0, 50.0] if annotate else [25.0, 50.0])
    ax12.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax12.tick_params(labelsize=16)
    h, l = [(a + b) for a, b in zip(ax1.get_legend_handles_labels(), ax12.get_legend_handles_labels())]
    order_idx = [0, 1]
    lgd = ax1.legend([h[i] for i in order_idx], [l[i] for i in order_idx], fontsize=16, loc='upper right',
                     framealpha=1.0, ncol=2)


def get_ds_day(ds):
    x, y, t, m = ds
    mask_x = np.logical_and(x > d * init_args.hours_day, x < (d + 1) * init_args.hours_day)
    mask_t = np.logical_and(t > d * init_args.hours_day, t < (d + 1) * init_args.hours_day)
    return x[mask_x] % 24.0, y[mask_x], t[mask_t] % 24.0, m[mask_t]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_day_train", type=int, default=3)
    parser.add_argument("--hours_day", type=float, default=24.0)
    parser.add_argument("--n_day_test", type=int, default=0)
    parser.add_argument("--patient_ids", type=str, default='12,25,28')
    parser.add_argument('--use_time_corrections', action='store_true')
    parser.add_argument('--use_bias', action='store_true')
    parser.add_argument('--log_meals', action='store_true')
    parser.add_argument("--treatment_covariates", type=str, default='SUGAR,STARCH')
    parser.add_argument('--output_dir', type=str, default='figures.appendix.figure6')
    init_args = parser.parse_args()
    # init_dict = vars(init_args)
    # init_dict.update(GENERAL_PARAMS)
    # init_args = argparse.Namespace(**init_dict)
    init_args.patient_ids = [int(s) for s in init_args.patient_ids.split(',')]
    init_args.preprocess_actions = True
    init_args.domain = [0.0, 24.0]
    os.makedirs(init_args.output_dir, exist_ok=True)
    pids0 = [[15, 21, 27], [3, 18, 36]]
    pids1 = [[4, 16, 19], [4, 19, 46]]
    pids2 = [[2, 17, 20], [5, 8, 17]]
    for pidx, sample_patient_ids in zip(init_args.patient_ids, [pids0, pids1, pids2]):
        plot_samples(pidx, sample_patient_ids, init_args)

    df_meta = get_updated_meta_df()
    models, f_all, ds_trains, ds_alls = [], [], [], []
    periods, data_slices = ['Baseline', 'Operation'], [1, 2]
    for period, data_slice in zip(periods, data_slices):
        df_joint = get_joint_df(period)
        ds_train, ds_test, ds_all = get_patient_datasets(df_meta, df_joint, period, data_slice, init_args)
        ds_alls.append(ds_all)

    for i, pidx in enumerate(init_args.patient_ids):
        ds_all1, ds_all2 = ds_alls[0][i], ds_alls[1][i]
        for d in range(3):
            ds1_day, ds2_day = get_ds_day(ds_all1), get_ds_day(ds_all2)
            plt.figure(figsize=(15, 4))
            ax = plt.gca()
            plot_joint_data(ds1_day, ds2_day)
            plt.tight_layout()
            plt.savefig(os.path.join(init_args.output_dir, f'appendix_fig6_real_data_p{pidx}_d{d}.pdf'))
            plt.close()
