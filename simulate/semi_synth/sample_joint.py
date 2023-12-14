import numpy as np

from models.mediator.gprpp.utils.tm_utils import get_relative_input_by_query


def sample_joint_tuples(patient_sampler, period, outcome_pidx, n_day, args):
    n_patient_out_oracle = len(args.oracle_outcome_pid)
    T = args.hours_day * n_day
    args.No = args.n_outcome * n_day
    noise_std = 0.1
    noise = np.random.randn(args.No) * noise_std
    x = np.linspace(0.0, T, args.No) if args.regular_measurement_times else np.sort(np.random.uniform(0.0, T, args.No))
    x = x.astype(np.float64)
    #
    action_time_sampler, action_mark_sampler = patient_sampler[0]
    outcome_sampler = patient_sampler[1]
    lambda_fnc = action_time_sampler.predict_lambda_compiled
    fb_mean, fb_var = outcome_sampler.predict_baseline_compiled([x.reshape(-1, 1) for _ in range(n_patient_out_oracle)])
    fb_mean = fb_mean.numpy().flatten()[args.No*outcome_pidx:args.No*(outcome_pidx + 1)]
    fb_var = fb_var.numpy().flatten()[args.No*outcome_pidx:args.No*(outcome_pidx + 1)]
    ft_mean, ft_var = np.zeros_like(x), np.zeros_like(x)
    f = fb_mean + ft_mean
    y = f + noise
    outcome_tuple = np.stack([x, y]).T
    f_median = np.median(f)
    outcome_tuple_norm = np.copy(outcome_tuple)
    outcome_tuple_norm[:, 1] -= f_median
    #
    current_t = x[0]
    until_t = get_closest_future_time(current_t, x, T)
    action_day, mark_day_activity_observed, mark_day_activity_effective, action_marked = [], [], [], np.zeros((0, 2))
    while current_t < T:
        ti, accepted = thinning_interval(lambda_fnc, current_t, until_t, np.array(action_day), outcome_tuple_norm, args)
        if len(ti) == 0:
            current_t = until_t
            until_t = get_closest_future_time(current_t, x, T)
        else:
            if accepted:
                action_day.append(ti.item())
                mark, _ = action_mark_sampler.predict_f_compiled(np.array([[ti[0] % 24.0]]))
                mark = mark.numpy().item()
                # Hidden Confounder, Activity Level around Meal, Affects Both Meal Amount and Glucose Response
                # Variables, M: mark, Z: activity, Y: response
                # M = mark^{z+1} / (z+1)
                # Y = alpha/(z+1) * M = alpha/(z+1)^2 * mark^{z+1}
                # if args.prob_activity > 0.0:
                activity = float(np.random.binomial(1, args.prob_activity)) if period == 'Operation' else 0.0
                mark = (mark ** (activity+1.0)) / (activity+1.0)
                mark_day_activity_observed.append(mark)
                mark_activity = mark / (2.0 ** activity)
                mark_day_activity_effective.append(mark_activity)
                # # Observed Action Mark
                action_marked = np.stack([action_day, mark_day_activity_observed]).T.astype(np.float64)
                # # True/Hidden Action Mark with Activity Effect
                action_marked_activity = np.stack([action_day, mark_day_activity_effective]).T.astype(np.float64)
                # assert np.allclose(action_marked_activity, action_marked)
                y, f, ft_mean, ft_var = update_outcome_tuple(outcome_sampler,
                                                             [x.reshape(-1, 1) for _ in range(n_patient_out_oracle)],
                                                             [action_marked_activity
                                                              for _ in range(n_patient_out_oracle)],
                                                             fb_mean, noise, outcome_pidx, args)
                outcome_tuple = np.stack([x, y]).T
                outcome_tuple_norm = np.copy(outcome_tuple)
                outcome_tuple_norm[:, 1] -= f_median
            current_t = ti.item()
            until_t = get_closest_future_time(current_t, x, T)

    f_means = [fb_mean, ft_mean, fb_mean+ft_mean]
    f_vars = [fb_var, ft_var, fb_var+ft_var]
    return action_marked, outcome_tuple, f_means, f_vars, f_median


def get_closest_future_time(current_t, xd, max_time):
    larger_event_idx = np.where(xd > current_t)[0]
    if len(larger_event_idx) > 0:
        smallest_larger_idx = np.min(larger_event_idx)
        return xd[smallest_larger_idx]
    else:
        return max_time


def thinning_interval(lambda_fnc, t1, t2, action_time, outcome_tuple, args):
    """

    :param lambda_fnc:
    :param t1:
    :param t2:
    :param action_time: (N,)
    :param outcome_tuple: (N, 2)
    :param args:
    :return:
    """
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


def update_outcome_tuple(outcome_sampler, xs, actions, fb, noise, patient_idx, args):
    t_lengths = [ti.shape[0] for ti in actions]
    Np = len(t_lengths)
    patient_order_arr = np.arange(Np, dtype=np.int32)
    tnew_patient_idx = np.repeat(patient_order_arr, t_lengths)
    ft_mean, ft_var = outcome_sampler.predict_ft_w_tnew_compiled(xs, actions, tnew_patient_idx)
    ft_mean = ft_mean.numpy().flatten()[args.n_outcome*patient_idx:args.n_outcome*(patient_idx+1)]
    ft_var = np.diag(ft_var.numpy()[0])[args.n_outcome*patient_idx:args.n_outcome*(patient_idx+1)]
    f = fb + ft_mean
    y = f + noise
    return y, f, ft_mean, ft_var
