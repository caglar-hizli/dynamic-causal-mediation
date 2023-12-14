import os
from argparse import Namespace

import tensorflow as tf

from utils.om_utils import prepare_om_output_dir
from train.outcome.compare_om_shared_marked_hier import train_ft_glucose_period
from train.mediator.run_mm_ind import train_tm_ind
from models.mediator.gprpp.utils.tm_utils import prepare_tm_output_dir
from utils.constants import PARAMS_GENERAL, PARAMS_TRAIN_OUTCOME_SAMPLER, PARAMS_TRAIN_TREATMENT_SAMPLER, \
    PARAMS_TRAIN_BASELINE, PARAMS_TRAIN_OPERATION


def get_oracle_sampler(period, args):
    treatment_model_str = ','.join([str(p) for p in args.oracle_treatment_pid])
    outcome_model_str = ','.join([str(p) for p in args.oracle_outcome_pid])
    treatment_model_dir = os.path.join(args.sampler_dir, period, f'treatment.p{treatment_model_str}')
    outcome_model_dir = os.path.join(args.sampler_dir, period, f'outcome.p{outcome_model_str}')
    if not os.path.exists(os.path.join(treatment_model_dir, 'time_intensity')):
        train_treatment_sampler(args, period)
    if not os.path.exists(os.path.join(outcome_model_dir, 'outcome_model')):
        train_outcome_sampler(args, period)
    treatment_sampler = [tf.saved_model.load(os.path.join(treatment_model_dir, model_name))
                         for model_name in ['time_intensity', 'mark_intensity']]
    outcome_sampler = tf.saved_model.load(os.path.join(outcome_model_dir, 'outcome_model'))
    oracle_sampler = (treatment_sampler, outcome_sampler)
    return oracle_sampler


def train_treatment_sampler(args, period):
    train_param_dict = {}
    train_param_dict.update(PARAMS_GENERAL)
    params_train_period = PARAMS_TRAIN_BASELINE if period == 'Baseline' else PARAMS_TRAIN_OPERATION
    train_param_dict.update(params_train_period)
    train_param_dict.update(PARAMS_TRAIN_TREATMENT_SAMPLER)
    train_args_treatment = Namespace(**train_param_dict)
    train_args_treatment.output_dir = args.sampler_dir
    train_args_treatment.patient_ids = train_args_treatment.oracle_treatment_pid
    train_args_treatment.patient_id = train_args_treatment.patient_ids[0]
    train_args_treatment.tc_folder = args.tc_folder
    train_args_treatment = prepare_tm_output_dir(train_args_treatment, is_multiple=False)
    train_tm_ind(train_args_treatment)


def train_outcome_sampler(args, period):
    train_param_dict = {}
    train_param_dict.update(PARAMS_GENERAL)
    params_train_period = PARAMS_TRAIN_BASELINE if period == 'Baseline' else PARAMS_TRAIN_OPERATION
    train_param_dict.update(params_train_period)
    train_param_dict.update(PARAMS_TRAIN_OUTCOME_SAMPLER)
    train_args_outcome = Namespace(**train_param_dict)
    train_args_outcome.oracle_outcome_pid = args.oracle_outcome_pid
    train_args_outcome.output_dir = args.sampler_dir
    train_args_outcome.patient_ids = train_args_outcome.oracle_outcome_pid
    train_args_outcome.tc_folder = args.tc_folder
    prepare_om_output_dir(train_args_outcome, True)
    train_ft_glucose_period(train_args_outcome)
