#!/bin/bash
#SBATCH -J audio_4gpu
#SBATCH -A m4431_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 128
#SBATCH --gpus-per-node=4
#SBATCH -t 01:00:00
#SBATCH -o logs/train_4gpu_%j.out
#SBATCH -e logs/train_4gpu_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

source ../dist_dl/bin/activate

export OMP_NUM_THREADS=4

echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "GPU configuration: 4 GPUs"

nvidia-smi

# Monitor GPU utilization every second
nvidia-smi \
    --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total \
    --format=csv \
    -l 1 > "logs/gpu_4gpu_${SLURM_JOB_ID}.csv" &

MONITOR_PID=$!

trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

# Run distributed training
srun torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    train.py \
    --batch_size 32 \
    --epochs 10 \
    --learning_rate 0.0003 \
    --num_workers 4 \
    --out_dir checkpoints/4gpu

echo "End time: $(date)"