#!/bin/bash
#SBATCH --job-name=audio-lr50
#SBATCH --account=m4431_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --array=0-5%2
#SBATCH --output=audio-lr50-%A_%a.out

set -euo pipefail
cd "$SCRATCH/audio-super-resolution"

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MODEL_LAYERS=6
export MODEL_WIDTH=256

case "$SLURM_ARRAY_TASK_ID" in
    0) export EXPERIMENT_LR=0.0001 MODEL_SEED=47 ;;
    1) export EXPERIMENT_LR=0.0003 MODEL_SEED=47 ;;
    2) export EXPERIMENT_LR=0.0010 MODEL_SEED=47 ;;
    3) export EXPERIMENT_LR=0.0001 MODEL_SEED=48 ;;
    4) export EXPERIMENT_LR=0.0003 MODEL_SEED=48 ;;
    5) export EXPERIMENT_LR=0.0010 MODEL_SEED=48 ;;
esac

echo "Learning rate=$EXPERIMENT_LR Seed=$MODEL_SEED"
srun "$SCRATCH/perlmutter-class/dist_dl/bin/python" \
    -u train_audio_lr50.py
