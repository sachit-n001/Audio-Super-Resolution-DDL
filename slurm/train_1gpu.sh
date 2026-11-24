#!/bin/bash
#SBATCH -J audio_1gpu
#SBATCH -A m4431_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus=1
#SBATCH -t 01:00:00
#SBATCH -o logs/train_1gpu_%j.out
#SBATCH -e logs/train_1gpu_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

source ../dist_dl/bin/activate

export OMP_NUM_THREADS=4

echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "GPU configuration: 1 GPU"

nvidia-smi

# Monitor GPU utilization every second
nvidia-smi \
    --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total \
    --format=csv \
    -l 1 > "logs/gpu_1gpu_${SLURM_JOB_ID}.csv" &

MONITOR_PID=$!

trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

# Run training
srun python -u train.py \
    --batch_size 32 \
    --epochs 10 \
    --learning_rate 0.0003 \
    --num_workers 4 \
    --out_dir checkpoints/1gpu

echo "End time: $(date)"