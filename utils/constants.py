ALL_PATIENT_IDS = sorted([9, 18, 26, 29, 32, 57, 60] + [12, 23, 25, 28, 31, 46, 63, 76])
PARAMS_GENERAL = {
    'T_treatment': 3.0,
    'hours_day': 24.0,
    'dataset_path': 'dataset/public_dataset.csv',
    'treatment_covariates': ['SUGAR', 'STARCH'],
    'maxiter': 1000,
    'outcome_maxiter': 500,
    'periods': ['Baseline', 'Operation'],
    'data_slices': [1, 2],
    'plot_interventional_trajectories': True,
    'hua_treatment_prune_threshold': 4.0,
    'threshold_hypo': 3.9,
    'threshold_hyper': 5.6,
}
# SAMPLER PARAMETERS
PARAMS_TRAIN_BASELINE = {
    'period': 'Baseline',
    'data_slice': 1,
}
PARAMS_TRAIN_OPERATION = {
    'period': 'Operation',
    'data_slice': 2,
}
PARAMS_TRAIN_OUTCOME_SAMPLER = {
    'n_day_train': 3,
    'n_day_test': 0,
    'use_time_corrections': True,
    'use_bias': True,
    'log_meals': True,
    'normalize_outcome': False,
}
PARAMS_TRAIN_TREATMENT_SAMPLER = {
    'oracle_treatment_pid': [12],
    'n_day_train': 3,
    'n_day_test': 0,
    'tm_components': 'ao',
    'marked': True,
    'Dt': 2,
    'D': 3,
    'time_dims': [0, 1],
    'mark_dims': [2],
    'marked_dt': [1],
    'treatment_dim': 0,
    'outcome_dim': 1,
    'M_times': 20,
    'variance_init': [0.1, 0.1],
    'lengthscales_init': [1.0, 100.0, 2.5],
    'variance_prior': [0.1, 0.1],
    'lengthscales_prior': [1.0, 100.0, 2.5],
    'domain': [0.0, 24.0],
    'beta0': 0.1,
    'remove_night_time': True,
    'preprocess_treatments': False,
    'use_time_corrections': False,
    'normalize_outcome': True,
    'log_meals': True,
    'share_variance': False,
    'share_lengthscales': False,
    'optimize_variance': False,
    'optimize_lengthscales': False,
}
PARAMS_SAMPLING_ALG = {
    'sample_y_mark': False,
    'n_outcome': 40,
    'regular_measurement_times': True,
    'set_n_outcome': True,
    'set_deltax_outcome': False,
    'noise_std': 0.1,
    'oracle_treatment_pid': [12],
}
PARAMS_SAMPLING_RW = {
    'sample_y_mark': True,
    'domain': [0.0, 24.0],
    'deltax': 0.25,
    'set_n_outcome': False,
    'set_deltax_outcome': True,
    'regular_measurement_times': True,
    'noise_std': 0.01,
    'patient_ids': ALL_PATIENT_IDS,
    'oracle_outcome_pid': ALL_PATIENT_IDS,
    'oracle_treatment_pid': ALL_PATIENT_IDS,
    'oracle_treatment_n_patient': 1,
    # GPRPP
    'remove_night_time': True,
    'use_time_corrections': True,
    'log_meals': False,
    'meal_threshold_per_day': 1,
}
# GPRPP
GPRPP_PARAMS = {
    'domain': [0.0, 24.0],
    'normalize_outcome': True,
    'share_variance': False,
    'share_lengthscales': False,
    'optimize_variance': False,
    'optimize_lengthscales': False,
    'remove_night_time': False,
    'M_times': 20,
    'beta0': 0.1,
}
# GPRPP model specific
GPRPP_FAO = {
    'tm_components': 'ao',
    'Dt': 2,
    'marked_dt': [1],
    'D': 3,
    'marked': True,
    'variance_init': [0.1, 0.1],
    'lengthscales_init': [1.0, 100.0, 2.5],
    'action_dim': 0,
    'outcome_dim': 1,
    'time_dims': [0, 1],
    'mark_dims': [2],
}
