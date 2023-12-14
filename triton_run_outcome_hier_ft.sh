#!/bin/bash
#SBATCH --time=16:00:00
#SBATCH --mem=16G
#SBATCH --output=triton_run_%A_%a.out
#SBATCH --array=1-4

module load anaconda
module load tensorflow

case $SLURM_ARRAY_TASK_ID in
    1)
      srun python train/outcome/compare_om_shared_marked_hier.py --patient_ids 9,12,18,23,25,26,28,29,31,32,46,57,60,63,76 --output_dir out/outcome_model_rw/hier_params --n_day_train 5 --use_time_corrections --use_bias --T_treatment 3.0 --tc_folder out/time-correction-015
      ;;
    2)
      srun python train/outcome/compare_om_shared_marked.py --patient_ids 9,12,18,23,25,26,28,29,31,32,46,57,60,63,76 --output_dir out/outcome_model_rw/shared_params --n_day_train 5 --use_time_corrections --use_bias --T_treatment 3.0 --tc_folder out/time-correction-015
      ;;
esac