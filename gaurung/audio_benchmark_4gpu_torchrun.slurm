#!/bin/bash
#SBATCH --job-name=audio-bench-4
#SBATCH --account=m4431_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:15:00
#SBATCH --output=audio-bench-4-%j.out

set -euo pipefail
cd "$SCRATCH/audio-super-resolution"

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

srun --ntasks=1 --gpu-bind=none \
  "$SCRATCH/perlmutter-class/dist_dl/bin/python" \
  -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=4 \
  train_audio_benchmark_torchrun.py
