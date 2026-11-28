#!/bin/bash
#SBATCH --array=0-2%1
#SBATCH --job-name=audio-scale50-1
#SBATCH --account=m4431_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=audio-scale50-1gpu-%A_%a.out

set -euo pipefail
cd "$SCRATCH/audio-super-resolution"
srun "$SCRATCH/perlmutter-class/dist_dl/bin/python" -u train_audio_scaling50_1gpu.py
