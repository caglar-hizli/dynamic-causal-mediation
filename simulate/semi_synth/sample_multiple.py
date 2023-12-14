from collections import deque

import numpy as np

from models.benchmarks.fpca.model_fpca_from_mcmc import get_dosage_fpca
from models.benchmarks.schulam.simulations.common import predict_outcome_trajectory
from models.mediator.gprpp.utils.tm_utils import get_relative_input_by_query
from simulate.semi_synth.sample_joint import get_closest_future_time


def sample_multiple_trajectories(models_cf, model_types, pidx, n_exp_id,
                                 outcome_time_domain, treatment_time_domain, args):
    n_models = len(models_cf)
    n_model_per_exp = int(n_models // n_exp_id)
    tm_types, om_types = [model_type[0] for model_type in model_types], [model_type[1] for model_type in model_types]
    outcome_models = [m[1] for m in models_cf]
    time_intensities = [m[0][0] for m in models_cf]
    mark_intensities = [m[0][1] for m in models_cf]
    x = get_outcome_times(outcome_time_domain, args)
    fb_means = [
        get_fb_mean(om, om_type=om_type, x=x, patient_idx=pidx,
                    period='Baseline' if i % n_model_per_exp < int(n_model_per_exp // 2) else 'Operation', args=args)
        for i, (om_type, om) in enumerate(zip(om_types, outcome_models))
    ]
    outcomes = [np.stack([x, fb_mean]).T for fb_mean in fb_means]
    fb_medians = [np.median(fb_mean) for fb_mean in fb_means]
    outcomes_norm = [np.copy(outcome) for outcome in outcomes]
    for outcome_norm, fb_median in zip(outcomes_norm, fb_medians):
        outcome_norm[:, 1] -= fb_median
    #
    T = treatment_time_domain[-1]
    current_t = treatment_time_domain[0]
    until_t = get_closest_future_time(current_t, x, T)
    baseline_time = np.zeros((1, 1)) + current_t
    actions = [np.zeros((0, 2)).astype(np.float64) for _ in models_cf]
    interval = (current_t, until_t)
    algorithm_log = []
    ub_noise, acc_noise = np.random.uniform(size=500), np.random.uniform(size=500)
    ub_noise_q, acc_noise_q = deque([n for n in ub_noise]), deque([n for n in acc_noise])
    while interval[0] < T:
        # print(f'Sampling time interval [{interval[0]},{interval[1]}]')
        u1 = ub_noise_q.pop()
        u2 = acc_noise_q.pop()
        lambda_sup = [get_lambda_sup(time_intensity, tm_type, interval,
                                     baseline_time, action[:, 0], outcome, args)
                      for time_intensity, action, outcome, tm_type in
                      zip(time_intensities, actions, outcomes_norm, tm_types)]
        lambda_sup = np.max(lambda_sup)
        ti = current_t + -1 / lambda_sup * np.log(u1)
        if ti > until_t:
            candidate_point = (ti, False)
            current_t = until_t
            until_t = get_closest_future_time(current_t, x, T)
        else:
            candidate_point = (ti, True)
            ti = np.array([ti])
            accepted_oracle = [False] * n_model_per_exp  # TODO
            for i, (time_intensity, mark_intensity, outcome_model, om_type, tm_type) in \
                    enumerate(zip(time_intensities, mark_intensities, outcome_models, om_types, tm_types)):
                # TODO
                period = 'Baseline' if i % n_model_per_exp < int(n_model_per_exp // 2) else 'Operation'
                # tm_type = oracle corresponds to using the oracle mediators
                if tm_type == 'oracle':
                    accepted = accepted_oracle[i % n_model_per_exp]
                    actions[i] = np.copy(actions[i % n_model_per_exp])  # copy oracle actions
                else:
                    accepted = thinning_interval_pair(ti, u2, time_intensity, tm_type, lambda_sup, baseline_time,
                                                      actions[i][:, 0], outcomes_norm[i], args)

                    # i in [0,..., n_model_per_exp-1]  corresponds to the ground-truth sampler
                    if i < n_model_per_exp:
                        accepted_oracle[i] = accepted

                    if accepted:
                        # Posterior sampling task for Table X
                        if args.sample_y_mark:
                            mark_mean, mark_var = mark_intensity.predict_y_compiled(np.array([[ti.item() % args.hours_day]]))
                            mark = mark_mean + np.random.randn() * np.sqrt(mark_var)
                        else:
                            mark, _ = mark_intensity.predict_f_compiled(np.array([[ti.item() % args.hours_day]]))
                            if i < n_model_per_exp and period == 'Baseline':
                                activity = float(np.random.binomial(1, args.prob_activity))
                                mark = (mark ** (activity + 1.0)) / ((activity + 1.0) ** 2)

                        actions[i] = np.concatenate([actions[i], [[ti.item(), mark.numpy().item()]]])

                if accepted:
                    f_mean = get_f_mean(outcome_model, om_type=om_type, x=x, action=actions[i],
                                        patient_idx=pidx, fb_mean=fb_means[i],
                                        period=period, args=args)
                    outcomes[i] = np.stack([x, f_mean]).T
                    outcomes_norm = [np.copy(outcome) for outcome in outcomes]
                    for outcome_norm, fb_median in zip(outcomes_norm, fb_medians):
                        outcome_norm[:, 1] -= fb_median

            current_t = ti.item()
            until_t = get_closest_future_time(current_t, x, T)

        algorithm_log.append([interval, lambda_sup, candidate_point, [u1, u2]])
        interval = (current_t, until_t)
    return actions, outcomes, outcomes_norm, algorithm_log


def get_outcome_times(outcome_time_domain, args):
    if args.set_n_outcome:
        x = np.linspace(outcome_time_domain[0], outcome_time_domain[1], args.n_outcome).astype(np.float64)
    elif args.set_deltax_outcome:
        x = np.arange(outcome_time_domain[0], outcome_time_domain[1]+args.deltax, args.deltax).astype(np.float64)
    else:
        raise ValueError
    return x


def get_fb_mean(outcome_model, om_type, x, patient_idx, period, args):
    n_patient_model = len(args.oracle_outcome_pid) if om_type == 'oracle_gp' else args.n_patient
    pidx = patient_idx % n_patient_model
    if om_type == 'schulam':
        fb_mean = predict_outcome_trajectory(outcome_model, args.train_datasets_schulam[period],
                                             pidx, xnew=x % 24.0, tnew=[])
    elif om_type == 'fpca':
        # FPCA uses normalized time interval [0, 1]
        fb_mean = outcome_model.predict_baseline((x % 24.0) / 24.0, pidx)
    elif om_type == 'oracle_gp' or om_type == 'estimated_gp':
        fb_mean = get_gp_baseline(outcome_model, x, pidx, n_patient_model)
    else:
        raise ValueError('Wrong model type!')
    return fb_mean


def get_f_mean(outcome_model, om_type, x, action, patient_idx, fb_mean, period, args):
    n_patient_model = len(args.oracle_outcome_pid) if om_type == 'oracle_gp' else args.n_patient
    pidx = patient_idx % n_patient_model
    if om_type == 'schulam':
        f_mean = predict_outcome_trajectory(outcome_model, args.train_datasets_schulam[period],
                                            pidx, xnew=x % 24.0, tnew=action[:, 0] % 24.0)
    elif om_type == 'fpca':
        dosage = get_dosage_fpca(x, action, args)
        # FPCA uses normalized time interval [0, 1]
        f_mean = outcome_model.predict_outcome((x % 24.0) / 24.0, dosage, pidx)
    elif om_type == 'oracle_gp' or om_type == 'estimated_gp':
        f_mean = get_gp_f(outcome_model, x, action, fb_mean, pidx, n_patient_model)
    else:
        raise ValueError('Wrong model type!')
    return f_mean


def get_gp_f(outcome_model, x, action, fb_mean, patient_idx, n_patient_model):
    n_outcome = x.shape[0]
    t_lengths = [action.shape[0]] * n_patient_model
    Np = len(t_lengths)
    patient_order_arr = np.arange(Np, dtype=np.int32)
    tnew_patient_idx = np.repeat(patient_order_arr, t_lengths)
    # print(n_patient_model, n_outcome)
    # print(t_lengths)
    # print(tnew_patient_idx)
    ft_mean, _ = outcome_model.predict_ft_w_tnew_compiled([x.reshape(-1, 1) for _ in range(n_patient_model)],
                                                          [action for _ in range(n_patient_model)],
                                                          tnew_patient_idx)
    ft_mean = ft_mean.numpy().flatten()[n_outcome*patient_idx:n_outcome*(patient_idx+1)]
    f = fb_mean + ft_mean
    return f


def get_gp_baseline(outcome_model, x, pidx, n_patient_model):
    n_outcome = x.shape[0]
    fb_mean, _ = outcome_model.predict_baseline_compiled([x.reshape(-1, 1) for _ in range(n_patient_model)])
    fb_mean = fb_mean.numpy().flatten()[n_outcome*pidx:n_outcome*(pidx + 1)]
    return fb_mean


def thinning_interval_pair(ti, u2, time_intensity, time_intensity_type,
                           lambda_sup, baseline_time, action_time, outcome_tuple,
                           args, sample_rejected=False):
    lambda_ti = get_lambda_x(time_intensity, time_intensity_type, ti, baseline_time, action_time, outcome_tuple, args)
    lambda_ti = lambda_ti.item()
    intensity_val = (lambda_sup - lambda_ti) if sample_rejected else lambda_ti
    accept = u2 <= (intensity_val / lambda_sup)
    return accept


def get_lambda_sup(time_intensity, time_intensity_type, interval, baseline_time, action_time,
                   outcome_tuple, args):
    N = 40
    t1, t2 = interval
    xx = np.linspace(t1, t2, N+1)[1:]
    lambda_xx = get_lambda_x(time_intensity, time_intensity_type, xx, baseline_time, action_time, outcome_tuple, args)
    lambda_sup = np.max(lambda_xx)
    return lambda_sup


def get_lambda_x(time_intensity, time_intensity_type, xx, baseline_time, action_time, outcome_tuple, args):
    if time_intensity_type in ['gprpp', 'oracle']:
        X = get_relative_input_by_query(xx, baseline_time, action_time, outcome_tuple, args)
        lambdaX = time_intensity.predict_lambda_compiled(X)[1].numpy().flatten()
    elif time_intensity_type == 'vbpp':
        # VBPP is defined over [0, 24], assuming exchangeable days
        lambdaX = time_intensity.predict_lambda_compiled(xx.reshape(-1, 1) % args.hours_day)[1].numpy().flatten()
    else:
        raise ValueError('Wrong treatment model type!')
    return lambdaX
