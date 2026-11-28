#!/bin/bash
#SBATCH --job-name=audio-data50
#SBATCH --account=m4431_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=audio-data50-%j.out

set -euo pipefail
cd "$SCRATCH/audio-super-resolution"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

PYTHON="$SCRATCH/perlmutter-class/dist_dl/bin/python"

# Preparation runs on the allocated node before GPU training.
# This also avoids heavy archive processing on the login node.
srun "$PYTHON" -u prepare_audio_50songs.py
srun "$PYTHON" -u train_audio_50songs.py
