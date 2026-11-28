#!/bin/bash
#SBATCH --job-name=audio-final-test
#SBATCH --account=m4431_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=audio-final-test-%j.out

set -euo pipefail
cd "$SCRATCH/audio-super-resolution"

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

srun "$SCRATCH/perlmutter-class/dist_dl/bin/python" \
    -u prepare_audio_final_test.py

srun "$SCRATCH/perlmutter-class/dist_dl/bin/python" \
    -u evaluate_audio_final_test.py spectral
