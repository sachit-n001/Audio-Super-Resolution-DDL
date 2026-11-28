#!/bin/bash
#SBATCH --job-name=audio-capacity50
#SBATCH --account=m4431_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --array=0-5%2
#SBATCH --output=audio-capacity50-%A_%a.out

set -euo pipefail
cd "$SCRATCH/audio-super-resolution"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

case "$SLURM_ARRAY_TASK_ID" in
    0) export MODEL_LAYERS=3 MODEL_WIDTH=128 MODEL_SEED=45 ;;
    1) export MODEL_LAYERS=6 MODEL_WIDTH=256 MODEL_SEED=45 ;;
    2) export MODEL_LAYERS=8 MODEL_WIDTH=512 MODEL_SEED=45 ;;
    3) export MODEL_LAYERS=3 MODEL_WIDTH=128 MODEL_SEED=46 ;;
    4) export MODEL_LAYERS=6 MODEL_WIDTH=256 MODEL_SEED=46 ;;
    5) export MODEL_LAYERS=8 MODEL_WIDTH=512 MODEL_SEED=46 ;;
esac

echo "Layers=$MODEL_LAYERS Width=$MODEL_WIDTH Seed=$MODEL_SEED"
srun "$SCRATCH/perlmutter-class/dist_dl/bin/python" \
    -u train_audio_capacity50.py
