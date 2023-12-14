import numpy as np
import pandas as pd


class Trajectory:
    def __init__(self, w, basis):
        self.w = np.array(w)
        self.basis = basis

    def value(self, x):
        return np.dot(self.basis.design(x), self.w)


class ObservationTimes:
    def __init__(self, low, high, avg_n_obs):
        self.low = low
        self.high = high
        self.avg_n_obs = avg_n_obs

    def sample(self, rng):
        n_obs = 1 + rng.poisson(self.avg_n_obs - 1)
        return np.sort(rng.uniform(self.low, self.high, n_obs))


class TreatmentPolicy:
    def __init__(self, history_window, weight, bias, effect_window, effect):
        self.history_window = history_window
        self.weight = weight
        self.bias = bias
        self.effect_window = effect_window
        self.effect = effect

    def sample_treatment(self, y, t, rng):
        t0 = t[-1]
        time_to = t0 - t
        in_window = time_to <= self.history_window
        avg = np.mean(y[in_window])
        prob_rx = sigmoid(self.weight * avg + self.bias)
        return rng.binomial(1, prob_rx)

    def treat(self, y, t, treated, t0, stack):
        y_rx = np.array(y)
        t_rx = np.array(t)

        in_future = t > t0
        in_range = t <= (t0 + self.effect_window)
        treat = (in_future & in_range)

        if stack:
            y_rx += self.effect * treat
        else:
            y_rx += self.effect * (treat & (~treated))

        return y_rx, t_rx, (treated | treat)


def sample_trajectory(traj, obs_proc, ln_a, ln_l, noise_scale, rng):
    x = obs_proc.sample(rng)
    C = _ou_cov(x, x, ln_a, ln_l) + noise_scale**2 * np.eye(len(x))
    y = traj.value(x) + rng.multivariate_normal(np.zeros(len(x)), C)
    # y = traj.value(x) + noise_scale * rng.normal(size=len(x))
    return (y, x)


def _ou_cov(t1, t2, ln_a, ln_l):
    a = np.exp(ln_a)
    l = np.exp(ln_l)
    D = np.expand_dims(t1, 1) - np.expand_dims(t2, 0)
    return a * np.exp(-np.abs(D) / l)


def _matern_cov(t1, t2, ln_a, ln_l):
    a = np.exp(ln_a)
    l = np.exp(ln_l)
    D = (np.expand_dims(t1, 1) - np.expand_dims(t2, 0)) / l
    return a * (1 + np.sqrt(3) * D) * np.exp(-np.sqrt(3) * D)


def sample_trajectory_with_extra(traj, obs_proc, noise_scale, extra, rng):
    x = np.array(sorted(list(obs_proc.sample(rng)) + extra))
    y = traj.value(x) + noise_scale * rng.normal(size=len(x))
    return (y, x)


def treat_data_set(samples, policy, rng):
    return [treat_sample(y, t, policy, rng) for y, t in samples]


def make_treatment_func(policy, rng, stack=False):
    def treat(y, t):
        return treat_sample(y, t, policy, rng, stack)
    return treat


def mix_treat_data_set(samples, policy1, policy2, rng):
    treated = []

    for y, t in samples:
        flip = rng.randint(2)
        if flip: treated.append(treat_sample(y, t, policy1, rng))
        else:    treated.append(treat_sample(y, t, policy2, rng))

    return treated


def null_treat_data_set(samples):
    treated = []

    for y, t in samples:
        treated.append((y, (t, np.zeros(len(t)))))

    return treated


def treat_sample(y, t, policy, rng, stack=False):
    y_rx = np.array(y)
    t_rx = np.array(t)
    treated = np.zeros(len(t), dtype=bool)
    rx = np.zeros(len(t))

    for i, t0 in enumerate(t):
        rx[i] = policy.sample_treatment(y_rx[:(i+1)], t_rx[:(i+1)], rng)

        if rx[i] == 1:
            y_rx, t_rx, treated = policy.treat(y_rx, t_rx, treated, t0, stack)

    return y_rx, (t_rx, rx)


def hide_treatments(treated, time):
    hidden = []

    for y, (t, rx) in treated:
        h_rx = rx.copy()
        h_rx[t > time] = False
        hidden.append((y, (t, h_rx)))

    return hidden


def truncate_treat_data_set(samples, time, policy, rng, final=False):
    treated = []

    for y, t in samples:
        treated.append( truncate_treat_sample(y, t, time, policy, rng, final) )

    return treated


def truncate_treat_sample(y, t, time, policy, rng, final=False, stack=False):
    y_rx = np.array(y)
    t_rx = np.array(t)
    treated = np.zeros(len(t), dtype=bool)
    rx = np.zeros(len(t))

    for i, t0 in enumerate(t):
        if t0 > time:
            break

        rx[i] = policy.sample_treatment(y_rx[:(i+1)], t_rx[:(i+1)], rng)

        if rx[i] == 1:
            y_rx, t_rx, treated = policy.treat(y_rx, t_rx, treated, t0, stack)

    # Add a final treatment at time.
    if final:
        rx[t_rx == time] = 1.0
        y_rx, t_rx, treated = policy.treat(y_rx, t_rx, treated, time, stack)

    return y_rx, (t_rx, rx)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def predict_outcome_trajectory(model, samples, idx, xnew, tnew):
    # t, rx = x
    y, x = samples[idx]
    xnew_transformed = transform_input(xnew, tnew)
    y_hat = model.predict(xnew_transformed, y, x)
    return y_hat


def transform_input(xnew, tnew):
    treated = np.zeros(len(xnew))
    for ti in tnew:
        mask_treatment_effect = np.logical_and(xnew >= ti, xnew < ti + 3.0)
        treated[mask_treatment_effect] = 1
    return xnew, treated


def evaluate_model(model, samples, pred_times):
    results = []

    def xi(x, i):
        return x[0][i], x[1][i]

    for t0 in pred_times:
        for i, (y, x) in enumerate(samples):
            t, rx = x
            obs = t <= t0

            if sum(obs) == len(y):
                continue

            y_hat = model.predict(x, y[obs], xi(x, obs))

            results.append(pd.DataFrame(
                np.c_[
                    np.repeat(i, len(y)),
                    np.repeat(t0, len(y)),
                    obs,
                    t, y, y_hat
                ],
                columns = [
                    'sample_num',
                    'pred_time',
                    'observed',
                    't', 'y', 'y_hat'
                ]
            ))

    return pd.concat(results, axis=0)
