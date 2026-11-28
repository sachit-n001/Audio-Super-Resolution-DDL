#!/bin/bash
#SBATCH --job-name=audio-bench-4
#SBATCH --account=m4431_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=00:15:00
#SBATCH --output=audio-bench-4-%j.out

set -euo pipefail
cd "$SCRATCH/audio-super-resolution"

export MASTER_ADDR="$(hostname)"
export MASTER_PORT="$((15000 + SLURM_JOB_ID % 40000))"
export OMP_NUM_THREADS=1

srun --gpus-per-task=1 --gpu-bind=single:1 \
  "$SCRATCH/perlmutter-class/dist_dl/bin/python" \
  -u train_audio_benchmark_ddp.py
