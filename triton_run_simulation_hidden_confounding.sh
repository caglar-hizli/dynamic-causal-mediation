#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --output=triton_run_%A_%a.out
#SBATCH --array=1-60

module load r
module load anaconda
module load tensorflow
pip install --user seaborn

# Job step
export SRDIR=out/sampler_semi_synth
export Np=50
export OPID=12,25,28
export BASEDIR=out/hidden_confounding

if [[ $SLURM_ARRAY_TASK_ID -gt 50 ]]
then
  export SDIR=${BASEDIR}/samples.100
  export MDIR=${BASEDIR}/models.100
  export pA=1.0
elif [[ $SLURM_ARRAY_TASK_ID -gt 40 ]]
then
  export SDIR=${BASEDIR}/samples.080
  export MDIR=${BASEDIR}/models.080
  export pA=0.8
elif [[ $SLURM_ARRAY_TASK_ID -gt 30 ]]
then
  export SDIR=${BASEDIR}/samples.060
  export MDIR=${BASEDIR}/models.060
  export pA=0.6
elif [[ $SLURM_ARRAY_TASK_ID -gt 20 ]]
then
  export SDIR=${BASEDIR}/samples.040
  export MDIR=${BASEDIR}/models.040
  export pA=0.4
elif [[ $SLURM_ARRAY_TASK_ID -gt 10 ]]
then
  export SDIR=${BASEDIR}/samples.020
  export MDIR=${BASEDIR}/models.020
  export pA=0.2
else
  export SDIR=${BASEDIR}/samples.000
  export MDIR=${BASEDIR}/models.000
  export pA=0.0
fi

srun python simulate/semi_synth/simulate_observational.py --sampler_dir ${SRDIR} --samples_dir ${SDIR} --oracle_outcome_pid_str ${OPID} --n_patient $Np --n_day 1 --seed $SLURM_ARRAY_TASK_ID --prob_activity $pA --tc_folder out/time-correction-015
srun Rscript models/benchmarks/fpca/R_script/simulation_fpca.R ${SDIR} Baseline $Np ${OPID} $SLURM_ARRAY_TASK_ID
srun Rscript models/benchmarks/fpca/R_script/simulation_fpca.R ${SDIR} Operation $Np ${OPID} $SLURM_ARRAY_TASK_ID
srun python simulate/semi_synth/run_simulation.py --task hidden-confounding --sampler_dir ${SRDIR} --samples_dir ${SDIR} --output_dir ${MDIR} --oracle_outcome_pid_str ${OPID} --n_patient $Np --n_day 1 --seed $SLURM_ARRAY_TASK_ID --prob_activity $pA --tc_folder out/time-correction-015
