#!/bin/bash
#SBATCH --job-name=audio-scaled
#SBATCH --account=m4431_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=audio-scaled-%j.out

set -e
cd "$SCRATCH/audio-super-resolution"
srun "$SCRATCH/perlmutter-class/dist_dl/bin/python" -u train_audio_scaled.py
