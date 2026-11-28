#!/bin/bash
#SBATCH --job-name=audio-conditioning50
#SBATCH --account=m4431_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --array=0-3%2
#SBATCH --output=audio-conditioning50-%A_%a.out

set -euo pipefail
cd "$SCRATCH/audio-super-resolution"

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MODEL_LAYERS=6
export MODEL_WIDTH=256

case "$SLURM_ARRAY_TASK_ID" in
  0) export MODEL_SEED=51 USE_CONDITION=1 ;;
  1) export MODEL_SEED=51 USE_CONDITION=0 ;;
  2) export MODEL_SEED=52 USE_CONDITION=1 ;;
  3) export MODEL_SEED=52 USE_CONDITION=0 ;;
esac

echo "Conditioning=$USE_CONDITION Seed=$MODEL_SEED"

srun "$SCRATCH/perlmutter-class/dist_dl/bin/python" \
    -u train_audio_conditioning50.py
