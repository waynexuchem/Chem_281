#!/bin/bash
#SBATCH -A m4164
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 5
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1

export SLURM_CPU_BIND="cores"
srun ./gpuintro
