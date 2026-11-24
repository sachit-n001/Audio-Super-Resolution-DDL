#!/bin/bash
#SBATCH -J audio_8gpu
#SBATCH -A m4431_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH --gpus-per-node=4
#SBATCH -t 01:00:00
#SBATCH -o logs/train_8gpu_%j.out
#SBATCH -e logs/train_8gpu_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

source ../dist_dl/bin/activate

export OMP_NUM_THREADS=4

# Configure communication between nodes
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "GPU configuration: 8 GPUs"
echo "Master node: $MASTER_ADDR"

# Launch one torchrun process per node
srun bash -c '
    nvidia-smi

    # Monitor GPU utilization on each node
    nvidia-smi \
        --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total \
        --format=csv \
        -l 1 > "logs/gpu_8gpu_${SLURM_JOB_ID}_$(hostname).csv" &

    MONITOR_PID=$!

    trap "kill $MONITOR_PID 2>/dev/null || true" EXIT

    torchrun \
        --nnodes=2 \
        --nproc_per_node=4 \
        --node_rank="$SLURM_NODEID" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        train.py \
        --batch_size 32 \
        --epochs 10 \
        --learning_rate 0.0003 \
        --num_workers 4 \
        --out_dir checkpoints/8gpu
'

echo "End time: $(date)"