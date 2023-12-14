#!/bin/bash -l
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=triton_log_%A_%a.out
#SBATCH --array=1-4

module load r/4.1.1-python3

case $SLURM_ARRAY_TASK_ID in
    1)
      srun Rscript --vanilla ./time-correction/SimpleModelHier.R Operation 015
      ;;
    2)
      srun Rscript --vanilla ./time-correction/SimpleModelHier.R Operation 100
      ;;
    3)
      srun Rscript --vanilla ./time-correction/SimpleModelHier.R Baseline 015
      ;;
    4)
      srun Rscript --vanilla ./time-correction/SimpleModelHier.R Baseline 100
      ;;
esac
