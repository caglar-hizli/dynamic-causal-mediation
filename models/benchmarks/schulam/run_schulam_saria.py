import logging
import os

import numpy as np
import matplotlib.pyplot as plt

from models.benchmarks.schulam.bsplines import BSplines
from models.benchmarks.schulam.simulations import common
from models.benchmarks.schulam.model import bspline_mixture_rx


def train_schulam(patient_datasets, high, args):
    low, high, n_bases, degree = 0.0, high, 5, 2
    basis = BSplines(low, high, n_bases, degree, boundaries='space')
    # sample_path = os.path.join('samples/joint.seed1/np20.ntr2.pa4,12.po2,3,8,12.no40.tTrue', f'patients.npz')
    # dataset_dict = np.load(sample_path, allow_pickle=True)
    # true_f_means = dataset_dict['true_f_means']
    # patient_datasets = dataset_dict['ds']
    train_ds = []
    for i in range(args.n_patient):
        ds = patient_datasets[i]
        x, y, t, m = ds[0], ds[1], ds[2], ds[3]
        x_transformed = common.transform_input(x, t)
        train_ds.append([y, x_transformed])

    n_classes, rx_duration = len(args.oracle_outcome_pid), 3.0
    model = bspline_mixture_rx.TreatmentBSplineMixture(basis, n_classes, rx_duration)
    model.fit(train_ds)
    return train_ds, model


def simulation():
    low, high, n_bases, degree = 0.0, 48.0, 5, 2
    prediction_times = [36.0]

    basis = BSplines(low, high, n_bases, degree, boundaries='space')
    print(basis._knots)

    sample_path = os.path.join('samples/joint.seed1/np20.ntr2.pa4,12.po2,3,8,12.no40.tTrue', f'patients.npz')
    dataset_dict = np.load(sample_path, allow_pickle=True)
    true_f_means = dataset_dict['true_f_means']
    patient_datasets = dataset_dict['ds']
    train_ds = []
    for i in range(20):
        ds = patient_datasets[i]
        x, y, t, m = ds[0], ds[1], ds[2], ds[3]
        x_transformed = common.transform_input(x, t)
        train_ds.append([y, x_transformed])

    # base1 = train_model(train1, basis, 4, 3.0)
    prop1 = train_model(train_ds, basis, 4, 3.0)
    prop_pred1 = common.evaluate_model(prop1, train_ds, prediction_times)
    for pidx in range(10):
        pred_pidx = prop_pred1[prop_pred1.sample_num == pidx]
        plt.plot(pred_pidx['t'], pred_pidx['y'], '-o', label='Y true')
        plt.plot(pred_pidx['t'], pred_pidx['y_hat'], '--', label='Y pred')
        plt.show()
    print('Done')


def train_model(data, basis, n_classes, rx_duration):
    model = bspline_mixture_rx.TreatmentBSplineMixture(basis, n_classes, rx_duration)
    model.fit(data)
    return model


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    simulation()
