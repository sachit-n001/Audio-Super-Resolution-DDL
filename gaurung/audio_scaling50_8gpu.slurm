#!/bin/bash
#SBATCH --array=0-2%1
#SBATCH --job-name=audio-scale50-8
#SBATCH --account=m4431_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:15:00
#SBATCH --output=audio-scale50-8gpu-%A_%a.out

set -euo pipefail
cd "$SCRATCH/audio-super-resolution"

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((15000 + SLURM_JOB_ID % 40000))

srun --ntasks=2 --ntasks-per-node=1 --gpu-bind=none \
  bash -c '
    exec "$SCRATCH/perlmutter-class/dist_dl/bin/python" \
      -m torch.distributed.run \
      --nnodes=2 \
      --nproc-per-node=4 \
      --node-rank="$SLURM_NODEID" \
      --master-addr="$MASTER_ADDR" \
      --master-port="$MASTER_PORT" \
      train_audio_scaling50_8gpu.py
  '
