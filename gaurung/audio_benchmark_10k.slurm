#!/bin/bash
#SBATCH --job-name=audio-diff-10k
#SBATCH --account=m4431_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=audio-diff-10k-%j.out

set -euo pipefail

cd "$SCRATCH/audio-super-resolution"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

srun "$SCRATCH/perlmutter-class/dist_dl/bin/python" \
  -u train_audio_benchmark_10k.py
