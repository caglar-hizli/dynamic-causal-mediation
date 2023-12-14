import os
import io
import msoffcrypto

import numpy as np
import pandas as pd

from models.mediator.gprpp.utils.tm_utils import remove_closeby_treatments

# From Ashrafi et al.
OAGB_IDS = [9, 18, 26, 29, 32, 57, 60]
RYGB_IDS = [12, 23, 25, 28, 31, 46, 63, 76]
PATIENT_IDS = sorted(OAGB_IDS+RYGB_IDS)


def get_joint_df(period):
    df_joint = pd.read_csv(f'data/data_{period}.csv')
    return df_joint

def get_updated_meta_df():
    df_meta = pd.read_csv(f'data/metadata_short.csv')
    return df_meta


def get_mediator_ds(df, pid, period, use_time_corrections=False, is_meal_logg=True, meal_threshold=0, tc_folder=None):
    x, y, t, m = get_outcome_ds(df, pid, period, use_time_corrections, is_meal_logg, tc_folder)
    x, y, t, m = remove_days_wo_treatment(x, y, t, m, meal_threshold)
    t, m = remove_closeby_treatments(t, m, threshold=0.5)
    t = np.sort(t)
    return x, y, t, m


def get_outcome_ds(df, pid, period, use_time_corrections=False, is_meal_logg=True, tc_folder=None):
    is_meal = df.y.isna()
    x = df.t[~is_meal].values
    day_offset = df.t.min() // 1440
    x = (x / 60) - day_offset * 24.0
    y = df.y[~is_meal].values
    df['MTOT'] = df.SUGAR + df.STARCH
    t = df.t[is_meal & (df.MTOT > 5)].values
    m = df.MTOT[is_meal & (df.MTOT > 5)].values
    m = np.log(m) if is_meal_logg else m
    t = (t / 60) - day_offset * 24.0
    t, m = remove_treatment_wo_effect(x, t, m)
    x, y, t, m = remove_days_wo_treatment(x, y, t, m, 0)
    if use_time_corrections:
        df_corr = pd.read_csv(os.path.join(tc_folder, period, 'time_corrections.csv'))
        tc = df_corr.values
        tidx = PATIENT_IDS.index(pid)
        nonzero_index, = np.nonzero(tc[tidx])
        t = tc[tidx, nonzero_index]
        t, m = remove_treatment_wo_effect(x, t, m)
    return x, y, t, m


def remove_treatment_wo_effect(x, t, m):
    remove_ids = []
    for i, ti in enumerate(t):
        if len(x[np.logical_and(x > ti, x < ti+1.0)]) == 0:
            remove_ids.append(i)
    t = np.delete(t, remove_ids)
    m = np.delete(m, remove_ids)
    return t, m


def remove_days_wo_treatment(x, y, t, m, meal_threshold=0):
    start_day, end_day = x.min() // 24, x.max() // 24
    # Remove days with less or equal to 1 meals
    remove_ds = []
    for d in range(int(start_day), int(end_day)+1):
        mask_t_day = np.logical_and(t >= d*24.0, t < (d+1)*24.0)
        if len(t[mask_t_day]) <= meal_threshold:
            remove_ds.append(d)

    mask_t = np.full_like(t, False, dtype=bool)
    mask_x = np.full_like(x, False, dtype=bool)
    for d in remove_ds:
        mask_t_day = np.logical_and(t >= d * 24.0, t < (d+1) * 24.0)
        mask_t = np.logical_or(mask_t, mask_t_day)
        mask_x_day = np.logical_and(x >= d * 24.0, x < (d+1) * 24.0)
        mask_x = np.logical_or(mask_x, mask_x_day)
    t, m = t[~mask_t], m[~mask_t]
    x, y = x[~mask_x], y[~mask_x]
    time_correction_t = np.zeros_like(t)
    time_correction_x = np.zeros_like(x)
    for d in remove_ds:
        time_correction_t[t >= d*24.0] += 1
        time_correction_x[x >= d*24.0] += 1
    t -= time_correction_t * 24.0
    x -= time_correction_x * 24.0
    return x, y, t, m


def train_test_split(ds, n_day_train):
    x, y, t, m = ds
    t_train_end = n_day_train * 24.0
    idx_x_train = x <= t_train_end
    idx_t_train = t <= t_train_end
    patient_train = (x[idx_x_train], y[idx_x_train], t[idx_t_train], m[idx_t_train])
    patient_test = (x[~idx_x_train], y[~idx_x_train], t[~idx_t_train], m[~idx_t_train])
    return patient_train, patient_test


def get_data_slice(df_meta, df, pid, period, data_slice):
    df_patient = df[df.id == pid]
    # Divide Operation data into 2 data slices (first 3 days, after 3 days)
    if period == 'Operation':
        op_day = df_meta.loc[df_meta.ID == pid, 'OpDateDiff Days'].item()
        if data_slice == 1:
            df_patient = df_patient[df_patient.t <= (op_day + 4) * 1440]
        elif data_slice == 2:
            df_patient = df_patient[df_patient.t > (op_day + 4) * 1440]
        else:
            pass

    # Extract data slice, where meals are recorded
    if data_slice:
        df_patient_meal = df_patient[df_patient.y.isna()]
        meal_start_day, meal_end_day = df_patient_meal.t.min() // 1440, df_patient_meal.t.max() // 1440
        df_patient = df_patient[(df_patient.t >= meal_start_day * 1440) &
                                (df_patient.t <= (meal_end_day+1) * 1440)]
    return df_patient
