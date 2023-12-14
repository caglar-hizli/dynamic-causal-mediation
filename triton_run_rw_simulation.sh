#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --output=triton_run_%A_%a.out
#SBATCH --array=1-10

module load r
module load anaconda
module load tensorflow
pip install --user seaborn

export SRDIR=out/sampler_rw
export SDIR=out/counterfactual/samples

# Reproduce Sampler
srun python train/outcome/compare_om_shared_marked_hier.py --patient_ids 9,12,18,23,25,26,28,29,31,32,46,57,60,63,76 --output_dir ${SRDIR} --n_day_train 500 --use_time_corrections --use_bias --log_meals --T_treatment 3.0 --tc_folder out/time-correction-015
srun python train/mediator/run_mm_pooled.py --output_dir ${SRDIR} --patient_ids 9,12,18,23,25,26,28,29,31,32,46,57,60,63,76 --n_day_train 500 --period Compare --M_times 20 --variance_init 0.1 0.1 --lengthscales_init 1.5 100.0 5.0 --variance_prior 0.1 0.1 --lengthscales_prior 1.5 100.0 5.0 --beta0 0.1 --remove_night_time --tm_components ao --marked --Dt 2 --marked_dt 2 --treatment_dim 0 --outcome_dim 1 --maxiter 1000  --use_time_corrections --meal_threshold_per_day 1 --seed $SLURM_ARRAY_TASK_ID --log_meals --tc_folder out/time-correction-100
# Job step
srun python simulate/real_world/simulate_counterfactual.py --sampler_dir ${SRDIR} --samples_dir ${SDIR} --use_time_corrections --tc_folder out/time-correction-100 --seed $SLURM_ARRAY_TASK_ID