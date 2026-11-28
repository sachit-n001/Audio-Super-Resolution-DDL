#!/bin/bash
#SBATCH --job-name=audio-precision
#SBATCH --account=m4431_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --array=0-5%1
#SBATCH --output=audio-precision-%A_%a.out

set -euo pipefail
cd "$SCRATCH/audio-super-resolution"

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export EXPERIMENT_UPDATES=10000

case "$SLURM_ARRAY_TASK_ID" in
    0) export EXPERIMENT_PRECISION=fp32 EXPERIMENT_SEED=42 ;;
    1) export EXPERIMENT_PRECISION=bf16 EXPERIMENT_SEED=42 ;;
    2) export EXPERIMENT_PRECISION=fp32 EXPERIMENT_SEED=43 ;;
    3) export EXPERIMENT_PRECISION=bf16 EXPERIMENT_SEED=43 ;;
    4) export EXPERIMENT_PRECISION=fp32 EXPERIMENT_SEED=44 ;;
    5) export EXPERIMENT_PRECISION=bf16 EXPERIMENT_SEED=44 ;;
esac

echo "Precision=$EXPERIMENT_PRECISION Seed=$EXPERIMENT_SEED"
srun "$SCRATCH/perlmutter-class/dist_dl/bin/python" \
    -u train_audio_precision.py
