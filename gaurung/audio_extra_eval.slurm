#!/bin/bash
#SBATCH --job-name=audio-extra-eval
#SBATCH --account=m4431_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --output=audio-extra-eval-%j.out

set -euo pipefail
cd "$SCRATCH/audio-super-resolution"

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

srun "$SCRATCH/perlmutter-class/dist_dl/bin/python" \
    -u evaluate_audio_extra.py "$1"
