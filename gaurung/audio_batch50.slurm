#!/bin/bash
#SBATCH --job-name=audio-batch50
#SBATCH --account=m4431_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --array=0-5%2
#SBATCH --output=audio-batch50-%A_%a.out

set -euo pipefail
cd "$SCRATCH/audio-super-resolution"

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MODEL_LAYERS=6
export MODEL_WIDTH=256

case "$SLURM_ARRAY_TASK_ID" in
    0) export EXPERIMENT_BATCH=16 MODEL_SEED=49 ;;
    1) export EXPERIMENT_BATCH=32 MODEL_SEED=49 ;;
    2) export EXPERIMENT_BATCH=64 MODEL_SEED=49 ;;
    3) export EXPERIMENT_BATCH=16 MODEL_SEED=50 ;;
    4) export EXPERIMENT_BATCH=32 MODEL_SEED=50 ;;
    5) export EXPERIMENT_BATCH=64 MODEL_SEED=50 ;;
esac

echo "Batch=$EXPERIMENT_BATCH Seed=$MODEL_SEED"
srun "$SCRATCH/perlmutter-class/dist_dl/bin/python" \
    -u train_audio_batch50.py
