#!/bin/bash
#SBATCH --job-name=audio-utilization
#SBATCH --account=m4431_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=audio-utilization-%j.out

set -euo pipefail
cd "$SCRATCH/audio-super-resolution"

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export SLURM_ARRAY_TASK_ID=0

mkdir -p analysis/gpu_utilization
MONITOR_FILE="analysis/gpu_utilization/gpu_${SLURM_JOB_ID}.csv"

nvidia-smi \
  --query-gpu=timestamp,index,uuid,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
  --format=csv \
  -l 1 > "$MONITOR_FILE" &
MONITOR_PID=$!

trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

srun "$SCRATCH/perlmutter-class/dist_dl/bin/python" \
  -u train_audio_scaling50_1gpu.py

echo "GPU monitoring saved to $MONITOR_FILE"
