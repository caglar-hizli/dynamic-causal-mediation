import numpy as np

from .common import Trajectory
from .common import ObservationTimes


def sample_dataset(trajectories, obs_proc, noise_scale, policy, rng):
    untreated = []
    treated = []

    for i, traj in enumerate(trajectories):
        y, t = sample_trajectory(traj, obs_proc, noise_scale, rng)
        untreated.append((y, (t, np.zeros(len(t)))))

        y_rx, t_rx, is_rx = treat_sample(y, t, policy, rng)
        treated.append((y_rx, (t_rx, is_rx)))

    return untreated, treated


def sample_trajectory(traj, obs_proc, noise_scale, rng):
    x = obs_proc.sample(rng)
    y = traj.value(x) + noise_scale * rng.normal(size=len(x))
    return (y, x)


def treat_sample(y, t, policy, rng):
    y_rx = np.array(y)
    t_rx = np.array(t)
    rx = np.zeros(len(t))

    for i, t0 in enumerate(t):
        rx[i] = policy.sample_treatment(y_rx[:(i+1)], t_rx[:(i+1)], rng)

        if rx[i] == 1:
            y_rx, t_rx = policy.treat(y_rx, t_rx, t0)

    return y_rx, t_rx, rx


class TreatmentPolicy:
    def __init__(self, history_window, weight, effect_window, effect):
        self.history_window = history_window
        self.weight = weight
        self.effect_window = effect_window
        self.effect = effect

    def sample_treatment(self, y, t, rng):
        t0 = t[-1]
        time_to = t0 - t
        in_window = time_to <= self.history_window
        avg = np.mean(y[in_window])
        prob_rx = _sigmoid(self.weight * avg)
        return rng.binomial(1, prob_rx)

    def treat(self, y, t, t0):
        y_rx = np.array(y)
        t_rx = np.array(t)

        in_future = t > t0
        in_range = t <= (t0 + self.effect_window)
        y_rx += self.effect * (in_future & in_range)

        return y_rx, t_rx


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
