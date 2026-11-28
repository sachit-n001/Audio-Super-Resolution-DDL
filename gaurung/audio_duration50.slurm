#!/bin/bash
#SBATCH --job-name=audio-duration50
#SBATCH --account=m4431_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --array=0-1%2
#SBATCH --output=audio-duration50-%A_%a.out

set -euo pipefail
cd "$SCRATCH/audio-super-resolution"

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MODEL_LAYERS=3
export MODEL_WIDTH=128
export MODEL_SEED=$((45 + SLURM_ARRAY_TASK_ID))

srun "$SCRATCH/perlmutter-class/dist_dl/bin/python" \
    -u train_audio_duration50.py
