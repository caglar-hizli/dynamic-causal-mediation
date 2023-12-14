import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from utils.data_utils import get_updated_meta_df, get_joint_df, get_data_slice, PATIENT_IDS, get_outcome_ds


def create_data_files(args):
    # N: #glucose measurements
    # M: #meals
    # y: glucose
    # t: glucose time
    # x1: meal carbs (g)
    # tx: meal time
    # hypers:
    # trend_p: glucose median
    df_meta = get_updated_meta_df()
    df_joint = get_joint_df(args.period)
    P = len(PATIENT_IDS)
    N, M, trend_p = np.array([], dtype=np.int8), np.array([], dtype=np.int8), np.array([], dtype=np.int8)
    for pidx in PATIENT_IDS:
        df = get_data_slice(df_meta, df_joint, pidx, args.period, args.data_slice)
        xi, yi, ti, mi = get_outcome_ds(df, pidx, args.period, False, is_meal_logg=False)
        N = np.append(N, yi.shape[0])
        M = np.append(M, mi.shape[0])
        trend_p = np.append(trend_p, np.median(yi))
        print(f'Patient[{pidx}], N={yi.shape[0]}, M={mi.shape[0]}')

    N_max = max(N)
    M_max = max(M)
    y = np.zeros((P, N_max))
    t = np.zeros((P, N_max))
    x1 = np.array([])  # Storage for carbs
    x2 = np.array([])  # Storage for fat
    x = np.array([])
    tx = np.array([])

    output_dir = os.path.join(args.output_dir, args.period)
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    for i, pidx in enumerate(PATIENT_IDS):
        df = get_data_slice(df_meta, df_joint, pidx, args.period, args.data_slice)
        xi, yi, ti, mi = get_outcome_ds(df, pidx, args.period, False, is_meal_logg=False)
        y[i, :N[i]] = yi
        t[i, :N[i]] = xi
        plt.figure(figsize=(15, 6))
        plt.plot(xi, yi, 'bo')
        plt.bar(ti, mi/np.max(mi), bottom=2.0, width=0.5, alpha=0.5, edgecolor='black', color='red')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f'joint_data_p{pidx}.pdf'))
        plt.close()
        x1 = np.append(x1, mi)
        tx = np.append(tx, ti)

    PM = len(x1)
    file_params = open(os.path.join(args.output_dir, args.period, "params.txt"), "w")
    file_params.write('\n'.join([str(P), str(N_max), str(M_max), str(PM)]))
    file_params.close()

    np.savetxt(os.path.join(output_dir, "N.txt"), N, fmt=['%d'])
    np.savetxt(os.path.join(output_dir, "M.txt"), M, fmt=['%d'])
    np.savetxt(os.path.join(output_dir, "trend_p.txt"), trend_p)
    np.savetxt(os.path.join(output_dir, "y.txt"), y)
    np.savetxt(os.path.join(output_dir, "t.txt"), t)
    np.savetxt(os.path.join(output_dir, "x1.txt"), x1)
    np.savetxt(os.path.join(output_dir, "x2.txt"), x2)
    np.savetxt(os.path.join(output_dir, "x.txt"), x)
    np.savetxt(os.path.join(output_dir, "tx.txt"), tx)
    np.savetxt(os.path.join(output_dir, "patients.txt"), PATIENT_IDS, fmt=['%s'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='data/stan')
    parser.add_argument("--data_slice", type=int, default=0)
    parser.add_argument("--period", type=str, default='Compare')
    init_args = parser.parse_args()
    create_data_files(init_args)
