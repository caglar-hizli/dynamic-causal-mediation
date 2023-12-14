import argparse
import os
import numpy as np


def get_relative_input_by_query(query_points, baseline_time, action_time, outcome_tuple, args):
    Nq = query_points.shape[0]
    rel_tuple_at_all_points = np.full((Nq, args.D), np.inf)
    d = 0
    # Relative time to baseline == absolute time
    if 'b' in args.tm_components:
        rel_tuple_at_all_points[:, d] = query_points - baseline_time
        d += 1
    # Relative time to actions
    if 'a' in args.tm_components:
        for i, xi in enumerate(query_points):
            smaller_action_idx = np.where(action_time < xi)[0]
            if len(smaller_action_idx) > 0:
                largest_smaller_idx = np.max(smaller_action_idx)
                rel_tuple_at_all_points[i, d] = xi - action_time[largest_smaller_idx]
        d += 1

    if 'o' in args.tm_components:
        # Relative time to outcomes
        for i, xi in enumerate(query_points):
            smaller_outcome_idx = np.where(outcome_tuple[:, 0] < xi)[0]
            if len(smaller_outcome_idx) > 0:
                largest_smaller_idx = np.max(smaller_outcome_idx)
                rel_tuple_at_all_points[i, d:] = outcome_tuple[largest_smaller_idx, :]
                rel_tuple_at_all_points[i, d] = xi - rel_tuple_at_all_points[i, d]  # Relative time to outcome
        d += 1

    return rel_tuple_at_all_points


def get_relative_input_joint(action_times, outcome_tuples, baseline_times, args, D):
    T = args.domain[1]
    rel_at_actions = []
    rel_at_all_points = []
    abs_all_points = []
    for action_time, outcome_tuple, baseline_time in zip(action_times, outcome_tuples, baseline_times):
        # Relative times to events: needed for the data term
        outcome_time = outcome_tuple[:, 0]
        rel_tuple_at_action = get_relative_input_by_query(action_time, baseline_time, action_time, outcome_tuple, args)
        rel_at_actions.append(rel_tuple_at_action)
        # Relative times to all: needed for the integral term
        all_time_point = []
        if 'b' in args.tm_components:
            all_time_point.append(baseline_time.flatten())
        if 'a' in args.tm_components:
            all_time_point.append(action_time.flatten())
        if 'o' in args.tm_components:
            all_time_point.append(outcome_time.flatten())
        all_time_point = np.sort(np.concatenate(all_time_point))
        # all_time_point = np.sort(np.concatenate([baseline_time.flatten(),
        #                                          action_time.flatten(),
        #                                          outcome_time.flatten()]))
        rel_tuple_at_all_points = get_relative_input_by_query(all_time_point, baseline_time, action_time,
                                                              outcome_tuple, args)
        # Add last region to relative times to all points for the integral computation
        last_action_time = action_time[-1] if len(action_time) > 0 else 0.0
        last_region = []
        if 'b' in args.tm_components:
            last_region.append(T)
        if 'a' in args.tm_components:
            last_region.append(T - last_action_time)
        if 'o' in args.tm_components:
            last_region.append(T - outcome_tuple[-1, 0])
            last_region.append(outcome_tuple[-1, 1])
        last_region = np.array(last_region)
        rel_tuple_at_all_points_shifted = np.concatenate([rel_tuple_at_all_points[1:, :], last_region.reshape(1, -1)])
        rel_at_all_points.append(rel_tuple_at_all_points_shifted)
        abs_all_points.append(all_time_point.reshape(-1, 1))
    return rel_at_actions, rel_at_all_points, abs_all_points


def set_mark_domain(marks, args):
    marks = marks[np.isfinite(marks)]
    if len(marks) > 0:
        args.mark_domain = (np.quantile(marks, 0.0), np.quantile(marks, 1.0))
    else:
        raise ValueError('No marks!')
    return args


def set_time_domain(rel_events, args):
    if 'a' in args.tm_components:
        relative_times_for_treat_dim = np.concatenate([rel_at_all[:, args.treatment_dim] for rel_at_all in rel_events])
        relative_times_for_treat_dim = relative_times_for_treat_dim[np.isfinite(relative_times_for_treat_dim)]
        args.treatment_time_domain = (0.0, np.quantile(relative_times_for_treat_dim, 0.95))
    else:
        args.treatment_time_domain = args.domain

    return args


def remove_closeby_treatments(t, m, threshold=2.0):
    idx = 0
    while idx < len(t):
        ti = t[idx]
        future = t[t > ti]
        if len(future) > 0 and len(future[future - ti < threshold]) > 0:
            removal_idx = [np.argwhere(t == tf).item() for tf in future[future - ti < threshold]]
            t = np.delete(t, removal_idx, None)
            m = np.delete(m, removal_idx, None)
        idx += 1
    return t, m


def prepare_tm_output_dir(args, is_multiple=False):
    patient_str = 'treatment.p'
    patient_str += ','.join([str(p) for p in args.patient_ids]) if is_multiple else f'{args.patient_id}'
    args.patient_model_dir = os.path.join(args.output_dir, f'{args.period}', patient_str)
    args.model_figures_dir = os.path.join(args.patient_model_dir, 'figures')
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.patient_model_dir, exist_ok=True)
    os.makedirs(args.model_figures_dir, exist_ok=True)
    return args


def get_tm_label(args):
    if args.tm_components == 'bao':
        label = r'$\lambda_{bao}(t) = (\beta_0 + f_b(t) + f_a(t) + f_o(t))^2$'
    elif args.tm_components == 'ao':
        label = r'$\lambda_{ao}(t) = (\beta_0 + f_a(t) + f_o(t))^2$'
    elif args.tm_components == 'ba':
        label = r'$\lambda_{ba}(t) = (\beta_0 + f_b(t) + f_a(t))^2$'
    elif args.tm_components == 'a':
        label = r'$\lambda_{a}(t) = (\beta_0 + f_a(t))^2$'
    elif args.tm_components == 'bo':
        label = r'$\lambda_{bo}(t) = (\beta_0 + f_b(t) + f_o(t))^2$'
    elif args.tm_components == 'b':
        label = r'$\lambda_{b}(t) = (\beta_0 + f_b(t))^2$'
    else:
        raise ValueError('Wrong action component combination!')
    return label


def get_gprpp_run_args(period, tm_components, output_dir):
    if tm_components == 'ao':
        return get_tm_fao_run_args(period, tm_components, output_dir)
    elif tm_components == 'a':
        return get_tm_fa_run_args(period, tm_components, output_dir)
    else:
        raise ValueError('Get run args function not implemented!')


def get_vbpp_run_args(period, output_dir):
    vbpp_init_dict = {
        'period': period,
        'output_dir': output_dir,
        'patient_ids': [0],
        'patient_id': 0,
        'maxiter': 10000,
        'domain': [0.0, 24.0],
    }
    init_args = argparse.Namespace(**vbpp_init_dict)
    return init_args


def get_tm_fao_run_args(period, tm_components, output_dir):
    treatment_args = argparse.Namespace(
        output_dir=output_dir,
        period=period,
        T_treatment=3.0,
        patient_ids=[0],
        patient_id=0,
        n_day_train=200,
        M_times=20,
        variance_init=[0.1, 0.1],
        lengthscales_init=[1.0, 100.0, 2.5],
        variance_prior=[0.1, 0.1],
        lengthscales_prior=[1.0, 100.0, 2.5],
        beta0=0.1,
        tm_components=tm_components,
        marked=True,
        Dt=2,
        D=3,
        marked_dt=[1],
        treatment_dim=0,
        outcome_dim=1,
        remove_night_time=False,
        preprocess_treatments=False,
        normalize_outcome=True,
        domain=[0.0, 24.0],
        share_variance=False,
        share_lengthscales=False,
        optimize_variance=False,
        optimize_lengthscales=False,
        maxiter=1000,
    )
    treatment_args.time_dims = list(range(treatment_args.Dt))
    treatment_args.mark_dims = list(range(treatment_args.Dt, treatment_args.D))
    return treatment_args


def get_tm_fa_run_args(period, tm_components, output_dir):
    treatment_args = argparse.Namespace(
        output_dir=output_dir,
        period=period,
        T_treatment=3.0,
        patient_ids=[0],
        patient_id=0,
        n_day_train=5,
        save_sampler=False,
        M_times=20,
        variance_init=[0.5],
        lengthscales_init=[1.0],
        variance_prior=[0.5],
        lengthscales_prior=[1.0],
        beta0=0.4,
        tm_components=tm_components,
        marked=False,
        Dt=1,
        marked_dt=[],
        D=1,
        treatment_dim=0,
        outcome_dim=-1,
        preprocess_treatments=False,
        remove_night_time=False,
        domain=[0.0, 24.0],
        share_variance=False,
        share_lengthscales=False,
        optimize_variance=False,
        optimize_lengthscales=False,
        maxiter=10000,
    )
    treatment_args.time_dims = list(range(treatment_args.Dt))
    treatment_args.mark_dims = list(range(treatment_args.Dt, treatment_args.D))
    return treatment_args
