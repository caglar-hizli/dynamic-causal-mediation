import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

from plot.plot_om import plot_gp_pred
from models.mediator.gprpp.utils.tm_utils import get_tm_label, get_relative_input_by_query

import matplotlib as mpl


# mpl.rcParams['text.usetex'] = True


def plot_vbpp(events, X, lambda_mean, upper, lower, title, true_lambda_mean=None,
              plot_path=None, plot_data=False):
    plt.figure(figsize=(12, 6))
    plt.title(title, fontsize=30)
    plt.xlabel('Hours', fontsize=12)
    plt.xlabel(r'$\lambda(\cdot)$', fontsize=12)
    plt.xlim(X.min(), X.max())
    if true_lambda_mean is not None:
        plt.plot(X, true_lambda_mean, 'blue', lw=2, label='Lambda True')
    plt.plot(X, lambda_mean, 'red', lw=2, label='Lambda Pred')
    plt.fill_between(X.flatten(), lower, upper, color='red', alpha=0.2)
    if plot_data:
        cmap = plt.cm.get_cmap('Dark2')
        for d, ev in enumerate(events):
            if d < 7:
                plt.vlines(ev, np.zeros_like(ev), [np.max(upper) + 0.1] * len(ev),
                           linestyles='--',
                           label=f'Day {int(d)}',
                           colors=cmap(d))
            else:
                plt.vlines(ev, np.zeros_like(ev), [np.max(upper) + 0.1] * len(ev),
                           linestyles='--',
                           colors=cmap(7))
    plt.legend(fontsize=18, loc='upper left')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def plot_gprpp_results(baseline_times, actions, outcome_tuples, action_model, args, model_figures_dir,
                       oracle_model=None):
    plot_train_fit_joint(action_model, baseline_times, actions, outcome_tuples, model_figures_dir, args,
                         plot_confidence=False, oracle_model=oracle_model)
    if 'b' in args.tm_components:
        plot_trig_kernel_baseline_comp(action_model, actions, model_figures_dir, args, plot_confidence=False,
                                       oracle_model=oracle_model)
    if 'a' in args.tm_components:
        plot_trig_kernel_action_comp(action_model, model_figures_dir, args, plot_confidence=False,
                                     oracle_model=oracle_model)
    if 'o' in args.tm_components:
        plot_trig_kernel_outcome_comp(action_model, model_figures_dir, args, oracle_model=oracle_model)


def compare_gprpp_train_fits(time_models, mark_models, args):
    if 'a' in args.tm_components:
        compare_ga(time_models, labels=[r'$g_{m,pre}^{*}(\tau)$', r'$g_{m,post}^{*}(\tau)$'], args=args)
    if 'o' in args.tm_components:
        compare_go(time_models, labels=[r'$g_{o,pre}^{*}(\tau)$', r'$g_{o,post}^{*}(\tau)$'], args=args)
    labels = [r'$\lambda_{pre}(m \mid \tau)$', r'$\lambda_{op}(m \mid \tau)$']
    compare_mark_intensity(mark_models, labels=labels, args=args)


def plot_fm_pred(xs, ms, xnew, f_mean, f_var, path=None):
    plt.figure(figsize=(10, 4))
    plot_gp_pred(xnew, f_mean, f_var, label=r'$f_m(\cdot)$')
    plt.plot(xs, ms, 'x', ms=10, label=f'Mark')
    # plt.ylim(0, 5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def compare_fm_pred(action_models, args):
    plt.figure(figsize=(10, 4))
    Xnew = np.stack([np.linspace(0.0, 24.0, 50), np.zeros(50)]).astype(np.float64).T
    for model, label, color in zip(action_models,
                                   [r'$f_m(\cdot)$ Baseline', r'$f_m(\cdot)$ Operation'],
                                   ['tab:blue', 'tab:orange']):
        f_mean, f_var = model.predict_f_compiled(Xnew)
        plot_gp_pred(Xnew[:, 0].reshape(-1, 1), f_mean, f_var, color=color, label=label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.model_figures_dir, f'compare_fm.pdf'))
    plt.close()


def plot_train_fit_joint(model, baseline_times, actions, outcome_tuples, model_figures_dir, args,
                         plot_confidence=True, oracle_model=None):
    N_test = 200
    for d, (baseline_time, action_time, outcome_tuple) in enumerate(zip(baseline_times, actions, outcome_tuples)):
        X_abs = np.linspace(*args.domain, N_test)
        X = get_relative_input_by_query(X_abs, baseline_time, action_time, outcome_tuple, args)

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 4), gridspec_kw={'height_ratios': [2, 1],
                                                                                        'wspace': 0.0,
                                                                                        'hspace': 0.0})
        plt.subplot(2, 1, 1)
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.xlim(np.min(X_abs), np.max(X_abs))
        lw = 1.5
        label = get_tm_label(args)

        f_mean, lambda_mean = model.predict_lambda(X)
        plt.plot(X_abs, lambda_mean, 'b--', lw=lw, label=label)
        plt.plot(X_abs, f_mean+0.1, '--', color='red', lw=lw, label='f')
        Xa, Xo = np.copy(X), np.copy(X)
        Xa[:, 1:] = np.inf
        Xo[:, 0] = np.inf
        _, la_mean = model.predict_lambda(Xa)
        _, lo_mean = model.predict_lambda(Xo)
        plt.plot(X_abs, la_mean, '--', color='orange', lw=lw, label=r'$\lambda_a$')
        plt.plot(X_abs, lo_mean, '--', color='yellow', lw=lw, label=r'$\lambda_o$')
        if oracle_model is not None:
            _, oracle_lambda_mean = oracle_model.predict_lambda_compiled(X)
            plt.plot(X_abs, oracle_lambda_mean, 'g--', lw=lw, label=r'$\lambda_{oracle}$')

        _ = plt.xticks(np.linspace(0.0, 20.0, 5), fontsize=12)
        ylim = plt.gca().get_ylim()
        plt.vlines(action_time, *ylim, colors='red', linewidth=2.0, label=r'Treatments $\mathbf{a}$', zorder=-1)
        plt.vlines(outcome_tuple[:, 0], *ylim, colors='black', alpha=0.05)
        plt.ylabel(r'$\lambda(t)$', fontsize=16)
        plt.legend(loc='upper left', fontsize=12)

        plt.subplot(2, 1, 2)
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.xlim(np.min(X_abs), np.max(X_abs))
        plt.plot(outcome_tuple[:, 0], outcome_tuple[:, 1], 'kx', label=r"Outcomes $\mathbf{o}$")
        ylim = plt.gca().get_ylim()
        plt.vlines(outcome_tuple[:, 0], *ylim, colors='black', alpha=0.1)
        plt.ylabel(r'$y(t)$', fontsize=16)
        plt.xlabel(r'Time $t$', fontsize=16)
        plt.xticks([])
        plt.legend(loc='upper left', fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(model_figures_dir, f'train_fit_d{d}_c{str(plot_confidence)[0]}.pdf'))
        plt.close()


def plot_trig_kernel_baseline_comp(model, actions, model_figures_dir, args, plot_confidence=True, oracle_model=None):
    d = 0
    action = actions[d]
    X = np.full((100, args.D), np.inf, dtype=float)
    X_flat = np.linspace(*args.domain, 100)
    X[:, 0] = X_flat
    lambda_mean, lower, upper = model.predict_lambda_and_percentiles(X)
    lower = lower.numpy().flatten()
    upper = upper.numpy().flatten()
    f_mean, lambda_mean = model.predict_lambda(X)

    plt.figure(figsize=(12, 6))
    plt.xlim(np.min(X_flat), np.max(X_flat))
    # plt.plot(X_flat, lambda_mean, 'r--', label=r'$\lambda(t)$')
    plt.plot(X_flat, f_mean, 'g--', label=r'$f(t)$', alpha=0.2)
    if oracle_model is not None:
        f_mean_oracle, _ = oracle_model.predict_lambda_compiled(X)
        plt.plot(X_flat, f_mean, 'r--', label=r'$f_{oracle}$', alpha=0.2)

    if plot_confidence:
        plt.fill_between(X_flat, lower, upper, color='red', alpha=0.2, label='Confidence')

    _ = plt.xticks(np.linspace(*args.treatment_time_domain, 10), fontsize=12)
    ylim = plt.gca().get_ylim()
    plt.vlines(action, *ylim, colors='blue')
    # _ = plt.yticks(np.linspace(0.0, 0.2, 5), fontsize=12)
    plt.xlabel(r'Time, $t$', fontsize=16)
    plt.ylabel(r'$\lambda(t)$', fontsize=16)
    plt.title(r'Estimated $\lambda(t)$', fontsize=20)
    plt.legend(loc='upper left', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(model_figures_dir, f'lambda_est_d{d}_Dt0_c{str(plot_confidence)[0]}.pdf'))
    plt.close()


def compare_action_marks(models, times, marks, args):
    label_fs = 32
    sns.set_style("whitegrid")
    marks_bs, marks_op = np.concatenate([mi for mi in marks[0]]), np.concatenate([mi for mi in marks[1]])
    times_bs, times_op = np.concatenate([ti for ti in times[0]]), np.concatenate([ti for ti in times[1]])
    arrival_samples = []
    Nr, Ni = 5000, 20
    for time, model in zip(times, models):
        arrival_sample = []
        for _ in range(Nr):
            Xnew = np.sort(np.random.uniform(0.0, 24.0, Ni)).astype(np.float64).reshape(-1, 1)
            # Xnew = ti.astype(np.float64).reshape(-1, 1)
            # Xnew = np.linspace(0.0, 24.0, Ni).astype(np.float64).reshape(-1, 1)
            y_mean, y_var = model.predict_y_compiled_full_cov(Xnew)
            y_mean, y_var = y_mean.numpy(), y_var.numpy()[0]
            eps = np.random.randn(Ni).reshape(-1, 1)
            Ly = np.linalg.cholesky(y_var + 1e-6 * np.eye(y_var.shape[0]))
            samples = y_mean + Ly @ eps
            arrival_sample = np.concatenate([arrival_sample, samples.reshape(-1)])
        arrival_samples.append(arrival_sample)

    avg_carb_bs = marks_bs.mean()
    avg_carb_op = marks_op.mean()
    print(f'Average carb.'
          f'[Baseline]: {avg_carb_bs:.3f}, {np.exp(avg_carb_bs):.3f}'
          f'[Operation]: {avg_carb_op:.3f}, {np.exp(avg_carb_op):.3f}')
    print(f'Median carb.'
          f'[Baseline]: {np.median(marks_bs):.3f}, {np.exp(np.median(marks_bs)):.3f}'
          f'[Operation]: {np.median(marks_op):.3f}, {np.exp(np.median(marks_op)):.3f}')

    # if args.log_meals:
    #     marks_bs, marks_op = np.exp(marks_bs), np.exp(marks_op)
    #     arrival_samples = [np.exp(ai) for ai in arrival_samples]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True, gridspec_kw={'wspace': 0.0, 'hspace': 0.0})
    Xnew = np.linspace(0.0, 24.0, 50).astype(np.float64).reshape(-1, 1)
    for ax in (ax1, ax2):
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.tick_params(labelsize=22)
        ax.set_xlim(0.75, 6.25)
        # ax.set_ylim(0.0, 0.3)
        # ax.set_xticks(np.linspace(0.0, 9., 7))
    bold_titles = [r'\textbf{Pre-surgery}', r'\textbf{Post-surgery}']
    ax1.text(4.260, 1.208, bold_titles[0], color="black", fontsize=28,
             horizontalalignment="left",
             verticalalignment="top",
             linespacing=1.2,
             bbox=dict(boxstyle="round", fc="white", ec="black", pad=0.3, alpha=1.0))

    ax1.set_ylabel(r'Density', fontsize=label_fs)
    bins = np.linspace(0.0, np.max(np.concatenate([marks_bs, marks_op])), 50)
    ax1.hist(marks_bs, bins, color='k', edgecolor='k', alpha=0.2, label='Pre- $m$', density=True)
    sns.kdeplot(arrival_samples[0], clip=(1.0, 6.0), color='k', lw=4, linestyle="--", ax=ax1,
                label=r'Pre- Posterior \textsc{Kde}')
    ax1.set_ylim(0.0, 1.32)
    # ax1.axvline(avg_carb_bs, color='k', linestyle='dashed', linewidth=4, alpha=0.2)
    # ax1.axvline(arrival_samples[0].mean(), color='k', linestyle='dashed', linewidth=4)
    #
    ax2.hist(marks_op, bins, color='tab:blue', edgecolor='k', alpha=0.2, density=True, label='Post- $m$')
    sns.kdeplot(arrival_samples[1], clip=(1.0, 6.0), color='tab:blue', lw=4, linestyle="--", ax=ax2,
                label=r'Post- Posterior \textsc{Kde}')
    # ax2.axvline(avg_carb_op, color='tab:blue', linestyle='dashed', linewidth=4, alpha=0.2)
    # ax2.axvline(arrival_samples[1].mean(), color='tab:blue', linestyle='dashed', linewidth=4)
    ax2.set_ylabel(r'Density', fontsize=label_fs)
    ax2.set_xlabel(r'Carb. intake per meal, $m$ ($\log$ grams)', fontsize=label_fs)
    ax2.text(4.115, 1.088, bold_titles[1], color="black", fontsize=28,
             horizontalalignment="left",
             verticalalignment="top",
             linespacing=1.2,
             bbox=dict(boxstyle="round", fc="white", ec="black", pad=0.3, alpha=1.0))
    ax2.set_ylim(0.0, 1.32)
    ax2.invert_yaxis()
    #
    h, l = [(a + b) for a, b in zip(ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
    order_idx = [0, 2, 1, 3]
    ax2.legend([h[i] for i in order_idx], [l[i] for i in order_idx], fontsize=24, loc='lower right', framealpha=1.0,
               ncol=2, bbox_to_anchor=(1.0, -0.6))
    #
    plt.tight_layout()
    plt.savefig(os.path.join(args.model_figures_dir, 'compare_treatment_marks.pdf'))
    plt.close()


def compare_next_action_times(models, actions, args):
    label_fs = 32
    sns.set_style("whitegrid")
    arrival_times = []
    for action in actions:
        arrival_time = [ai[1:] - ai[:-1] for ai in action]
        arrival_times.append(np.concatenate(arrival_time))

    avg_arr_time_bs = arrival_times[0].mean()
    avg_arr_time_op = arrival_times[1].mean()
    print(f'Average next meal times, [Baseline]: {avg_arr_time_bs:.3f}, [Operation]: {avg_arr_time_op:.3f}')
    print(f'Median next meal times, [Baseline]: {np.median(arrival_times[0]):.3f}, '
          f'[Operation]: {np.median(arrival_times[1]):.3f}')

    arrival_samples = []
    Ns = 5000
    for arrival_time, model in zip(arrival_times, models):
        arrival_sample = []
        for i in range(Ns):
            arrival_sample += sample_single_event(model, args)
        arrival_samples.append(np.array(arrival_sample))

    bins = np.linspace(0.0, np.max(np.concatenate(arrival_samples)), 50)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True, gridspec_kw={'wspace': 0.0, 'hspace': 0.0})
    for ax in (ax1, ax2):
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.tick_params(labelsize=22)
        ax.set_xlim(-0.45, 9.45)
    ax1.tick_params(axis='x', which='both', length=0)
    ax2.set_xticks(np.linspace(0.0, 9., 7))
    # ax2.tick_params(axis='y', which='both', length=0)
    bold_titles = [r'\textbf{Pre-surgery}', r'\textbf{Post-surgery}']
    # ax1.set_xlabel(r'Next Meal Time (hours)', fontsize=24)

    # ax1.hist(arrival_times, bins, edgecolor='k', label='data', density=True)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(args.model_figures_dir, 'next_treatment_time_data.pdf'))
    # plt.close()
    print(f'Number of samples: {len(arrival_samples[0])}, {len(arrival_samples[1])}')
    ax1.hist(arrival_times[0], bins, color='k', edgecolor='k', alpha=0.2, density=True,
             label=r'Pre- $\Delta t$', )
    sns.kdeplot(arrival_samples[0], clip=(0.0, 9.0), color='k', lw=4, linestyle="--", ax=ax1,
                label=r'Pre- Posterior \textsc{Kde}')
    # ax1.axvline(avg_arr_time_bs, color='k', linestyle='--', linewidth=4, alpha=0.2)
    # ax1.axvline(arrival_samples[0].mean(), color='k', linestyle='--', linewidth=4)
    ax1.set_ylim(0.0, 0.55)
    ax1.set_yticks(np.linspace(0.0, 0.5, 6))
    ax1.text(5.865, 0.483, bold_titles[0], color="black", fontsize=28,
             horizontalalignment="left",
             verticalalignment="top",
             linespacing=1.2,
             bbox=dict(boxstyle="round", fc="white", ec="black", pad=0.3, alpha=1.0))
    ax1.set_ylabel(r'Density', fontsize=label_fs)
    #
    ax2.hist(arrival_times[1], bins, color='tab:blue', edgecolor='k', alpha=0.2, density=True,
             label=r'Post- $\Delta t$')
    sns.kdeplot(arrival_samples[1], clip=(0.0, 9.0), color='tab:blue', lw=4, linestyle="--", ax=ax2,
                label=r'Post- Posterior \textsc{Kde}')
    # ax2.axvline(avg_arr_time_op, color='tab:blue', linestyle='--', linewidth=4, alpha=0.2)
    # ax2.axvline(arrival_samples[1].mean(), color='tab:blue', linestyle='--', linewidth=4)
    ax2.set_xlabel(r'Next Meal Time, $\Delta t$ (hours)', fontsize=label_fs)
    ax2.set_ylabel(r'Density', fontsize=label_fs)
    ax2.text(5.600, 0.433, bold_titles[1], color="black", fontsize=28,
             horizontalalignment="left",
             verticalalignment="top",
             linespacing=1.2,
             bbox=dict(boxstyle="round", fc="white", ec="black", pad=0.3, alpha=1.0))
    ax2.set_ylim(0.0, 0.55)
    ax2.set_yticks(np.linspace(0.0, 0.5, 6))
    ax2.invert_yaxis()

    h, l = [(a + b) for a, b in zip(ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
    order_idx = [0, 2, 1, 3]
    ax2.legend([h[i] for i in order_idx], [l[i] for i in order_idx], fontsize=24, loc='lower right', framealpha=1.0,
               ncol=2, bbox_to_anchor=(1.015, -0.6))

    #
    # sns.hist(arrival_samples, bins, edgecolor='k', alpha=0.5, label='samples', density=True)
    # plt.axvline(arrival_samples.mean(), color='orange', linestyle='dashed', linewidth=2)
    # plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.savefig(os.path.join(args.model_figures_dir, 'compare_treatment_times.pdf'))
    plt.close()


def plot_next_action(model, actions, args):
    arrival_times = [ai[1:] - ai[:-1] for ai in actions]
    for i, ai in enumerate(arrival_times):
        print(i, ai)
    arrival_times = np.concatenate(arrival_times)

    N = len(arrival_times)
    print(f'Number of data points: {N}')
    arrival_samples = []
    for i in range(N):
        arrival_samples += sample_single_event(model, args)
    arrival_samples = np.array(arrival_samples)
    bins = np.linspace(0.0, np.max(arrival_samples), 30)
    plt.hist(arrival_times, bins, edgecolor='k', label='data', density=True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.model_figures_dir, 'next_treatment_time_data.pdf'))
    plt.close()
    print(f'Number of samples: {len(arrival_samples)}')
    plt.hist(arrival_times, bins, edgecolor='k', alpha=0.5, label='data', density=True)
    plt.hist(arrival_samples, bins, edgecolor='k', alpha=0.5, label='samples', density=True)
    plt.axvline(arrival_times.mean(), color='blue', linestyle='dashed', linewidth=2)
    plt.axvline(arrival_samples.mean(), color='orange', linestyle='dashed', linewidth=2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.model_figures_dir, 'next_treatment_time_samples_N.pdf'))
    plt.close()
    #
    for N in [500, 1000]:
        arrival_samples = []
        for i in range(N):
            arrival_samples += sample_single_event(model, args)
        arrival_samples = np.array(arrival_samples)
        bins = np.linspace(0.0, np.max(arrival_samples), 30)
        plt.hist(arrival_times, bins, edgecolor='k', alpha=0.5, label='data', density=True)
        plt.hist(arrival_samples, bins, edgecolor='k', alpha=0.5, label='samples', density=True)
        plt.axvline(arrival_times.mean(), color='blue', linestyle='dashed', linewidth=2)
        plt.axvline(arrival_samples.mean(), color='orange', linestyle='dashed', linewidth=2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.model_figures_dir, f'next_treatment_time_samples_{N}.pdf'))
        plt.close()


def sample_single_event(model, args, t_max=16.0):
    Nx = 20
    X = np.full((Nx, args.D), np.inf, dtype=float)

    t, interval = 0.0, 0.5
    while t < t_max:
        X_flat = np.linspace(t, t + interval, Nx)
        X[:, args.treatment_dim] = X_flat
        X[:, args.outcome_dim] = 0.1
        X[:, -1] = 0.0
        _, lambdaX = model.predict_lambda_compiled(X)
        lambda_sup = np.max(lambdaX)
        ti = t + np.random.exponential(1 / lambda_sup)
        if ti > t + interval:
            t = t + interval
        else:
            xi = np.array([[ti, 0.1, 0.0]])
            _, lambdaXi = model.predict_lambda_compiled(xi)
            lambda_vals = lambdaXi.numpy().flatten().item()
            accept = np.random.uniform(0.0, 1.0) <= (lambda_vals / lambda_sup)
            if accept:
                return [ti]
            t = ti
    return []


def thinning_interval(lambda_fnc, t1, t2, action_time, outcome_tuple, args):
    N = 20
    baseline_time = np.zeros((1, 1))
    xx = np.linspace(t1, t2, N+1)[1:]
    X = get_relative_input_by_query(xx, baseline_time, action_time, outcome_tuple, args)
    _, lambdaX = lambda_fnc(X)
    lambda_sup = np.max(lambdaX)
    ti = t1 + np.random.exponential(1 / lambda_sup, size=1)
    # n_points = np.random.poisson(lambda_sup * (t2 - t1))
    if ti.item() > t2:
        return [], False

    X = get_relative_input_by_query(ti, baseline_time, action_time, outcome_tuple, args)
    _, lambdaX = lambda_fnc(X)
    lambda_vals = lambdaX.numpy().flatten().item()
    accept = np.random.uniform(0.0, 1.0) <= (lambda_vals / lambda_sup)
    return ti, accept


def plot_trig_kernel_action_comp(model, model_figures_dir, args, plot_confidence=True, oracle_model=None):
    X = np.full((100, args.D), np.inf, dtype=float)
    X_flat = np.linspace(*args.treatment_time_domain, 100)
    X[:, args.treatment_dim] = X_flat
    f_mean, lambda_mean = model.predict_lambda(X)

    plt.figure(figsize=(12, 6))
    plt.xlim(np.min(X_flat), np.max(X_flat))
    # plt.plot(X_flat, lambda_mean, 'r--', label=r'$\lambda(t)$')
    plt.plot(X_flat, f_mean, 'g--', label=r'$f(t)$', alpha=0.2)
    if oracle_model is not None:
        f_mean_oracle, _ = oracle_model.predict_lambda_compiled(X)
        plt.plot(X_flat, f_mean_oracle, 'r--', label=r'$f_{oracle}$', alpha=0.2)

    if plot_confidence:
        _, lower, upper = model.predict_lambda_and_percentiles(X)
        lower = lower.numpy().flatten()
        upper = upper.numpy().flatten()
        plt.fill_between(X_flat, lower, upper, color='red', alpha=0.2, label='Confidence')

    _ = plt.xticks(np.linspace(*args.treatment_time_domain, 10), fontsize=12)
    plt.xlabel(r'Time, $t$', fontsize=16)
    plt.ylabel(r'$\lambda(t)$', fontsize=16)
    plt.title(r'Estimated $\lambda(t)$', fontsize=20)
    plt.legend(loc='upper right', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(model_figures_dir, f'lambda_est_Dt1_c{str(plot_confidence)[0]}.pdf'))
    plt.close()


def plot_trig_kernel_outcome_comp(model, model_figures_dir, args, oracle_model=None):
    N_test = 100
    t_grid = np.linspace(*args.treatment_time_domain, N_test + 1)
    m_grid = np.linspace(*args.mark_domain, N_test + 1)
    xx, yy = np.meshgrid(t_grid, m_grid)
    X_plot = np.vstack((xx.flatten(), yy.flatten())).T
    d_marked = args.marked_dt[0]
    X = np.full((X_plot.shape[0], args.D), np.inf, dtype=float)
    X[:, d_marked] = X_plot[:, 0]
    X[:, d_marked+1] = X_plot[:, 1]
    f_pred, lambda_pred = model.predict_lambda(X)
    lambda_pred_2d = lambda_pred.numpy().reshape(*xx.shape)
    f_pred_2d = f_pred.numpy().reshape(*xx.shape)

    plt.figure(figsize=(12, 6))
    mark_ids = np.linspace(0, N_test, 5).astype(int)
    for m_idx in mark_ids:
        m = yy[int(m_idx), 0]
        lambda_pred_m = lambda_pred_2d[int(m_idx)]
        plt.plot(t_grid, lambda_pred_m, lw=2, label=r'$\lambda(\cdot,$ ' + f'{m:.1f})')
    plt.title('Estimated Marked Intensity 1D', fontsize=24)
    plt.legend(fontsize=18, loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(model_figures_dir, f'lambda_est_marked_1d.pdf'))
    plt.close()

    plt.figure(figsize=(12, 6))
    mark_ids = np.linspace(0, N_test, 5).astype(int)
    for m_idx in mark_ids:
        m = yy[int(m_idx), 0]
        f_pred_m = f_pred_2d[int(m_idx)]
        plt.plot(t_grid, f_pred_m, lw=2, label=r'$f(\cdot,$ ' + f'{m:.1f})')

    plt.title('Estimated Marked Intensity 1D', fontsize=24)
    plt.legend(fontsize=18, loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(model_figures_dir, f'f_est_marked_1d.pdf'))
    plt.close()

    plt.figure(figsize=(12, 6))
    _ = plt.contourf(
        xx,
        yy,
        lambda_pred_2d,
        20,
        # [0.5],  # plot the p=0.5 contour line only
        cmap="RdGy_r",
        linewidths=2.0,
        # zorder=100,
    )
    plt.colorbar()
    plt.title('Estimated Marked Intensity 2D', fontsize=24)
    plt.savefig(os.path.join(model_figures_dir, f'lambda_est_marked_2d.pdf'))
    plt.close()

    X = np.full((N_test+1, args.D), np.inf, dtype=float)
    X[:, d_marked] = 12.0
    X[:, d_marked + 1] = m_grid
    f_pred, lambda_pred = model.predict_lambda(X)
    plt.figure(figsize=(12, 6))
    plt.plot(m_grid, f_pred, 'b', label=r'Estm. $f(12,\mathbf{m})$')
    if oracle_model is not None:
        f_mean_oracle, _ = oracle_model.predict_lambda_compiled(X)
        plt.plot(m_grid, f_mean_oracle, 'g--', label=r'$f_{oracle}$', alpha=0.2)

    plt.legend(fontsize=18, loc='upper right')
    plt.xlabel(r'Marks, $\mathbf{m}$', fontsize=18)
    plt.ylabel(r'$\lambda(\cdot, cdot)$', fontsize=18)
    plt.title('Estimated vs. True Mark Effect', fontsize=24)
    #
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.xticks(np.linspace(args.mark_domain[0], args.mark_domain[1], 10))
    plt.tight_layout()
    plt.savefig(os.path.join(model_figures_dir, f'mark_effect_est.pdf'))
    plt.close()


def compare_ga(models, labels, args):
    X = np.full((100, args.D), np.inf, dtype=float)
    X_flat = np.linspace(*args.treatment_time_domain, 100)
    X[:, args.treatment_dim] = X_flat

    plt.figure(figsize=(6, 4))
    plt.xlim(np.min(X_flat), np.max(X_flat))
    for label, model in zip(labels, models):
        f_mean, _ = model.predict_lambda_compiled(X)
        plt.plot(X_flat, f_mean, '--', linewidth=4, label=label, alpha=1.0)

    _ = plt.xticks(np.linspace(*args.treatment_time_domain, 10), fontsize=12)
    plt.xlabel(r'Time, $\tau$', fontsize=20)
    plt.ylabel(r'$g^*_m(\tau, \mathbf{m})$', fontsize=20)
    # plt.title(r'Ground-Truth $g^*_a(\tau)$', fontsize=18)
    plt.legend(loc='lower right', fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(args.model_figures_dir, f'compare_ga.pdf'))
    plt.close()


def compare_go(models, labels, args):
    N_test = 100
    t_grid = np.linspace(*args.domain, N_test + 1)
    m_grid = np.linspace(*args.mark_domain, N_test + 1)
    xx, yy = np.meshgrid(t_grid, m_grid)
    X_plot = np.vstack((xx.flatten(), yy.flatten())).T
    d_marked = args.marked_dt[0]
    X = np.full((X_plot.shape[0], args.D), np.inf, dtype=float)
    X[:, d_marked] = X_plot[:, 0]
    X[:, d_marked+1] = X_plot[:, 1]

    plt.figure(figsize=(6, 4))
    idx = int(N_test // 2)
    for label, model in zip(labels, models):
        f_pred, lambda_pred = model.predict_lambda_compiled(X)
        lambda_pred_2d = lambda_pred.numpy().reshape(*xx.shape)
        f_pred_2d = f_pred.numpy().reshape(*xx.shape)
        f_mark_effect = f_pred_2d[:, idx]
        plt.plot(m_grid, f_mark_effect, '--', linewidth=4, label=label)
    plt.legend(fontsize=16, loc='lower left')
    plt.xlabel(r'Glucose, $y$', fontsize=18)
    plt.ylabel(r'$g^*_o(\tau, \mathbf{y})$', fontsize=18)
    # plt.title(r'Ground-Truth $g^*_o(\tau)$', fontsize=18)
    #
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.xticks(np.linspace(args.mark_domain[0], args.mark_domain[1], 10))
    plt.tight_layout()
    plt.savefig(os.path.join(args.model_figures_dir, f'compare_go.pdf'))
    plt.close()


def compare_mark_intensity(models, labels, args):
    N_test = 100
    X_flat = np.linspace(0.0, 24.0, N_test)
    #
    plt.figure(figsize=(6, 4))
    # plt.xlim(*args.mark_domain)
    for label, model in zip(labels, models):
        f_mean, _ = model.predict_f_compiled(X_flat.reshape(-1, 1))
        f_mean = f_mean.numpy().flatten()
        plt.plot(X_flat, f_mean, '--', linewidth=4, label=label, alpha=1.0)

    _ = plt.xticks(np.linspace(0.0, 24.0, 10), fontsize=12)
    plt.ylabel(r'Carb. Intake (Dosage, $\log g$)', fontsize=18)
    plt.xlabel(r'Time, $\tau$', fontsize=18)
    # plt.title(r'Mark Intensity Functions $\lambda(m \mid \tau)$', fontsize=18)
    plt.legend(loc='lower right', fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(args.model_figures_dir, f'compare_mark_intensity.pdf'))
    plt.savefig(os.path.join(args.model_figures_dir, f'compare_mark_intensity.png'))
    plt.close()
