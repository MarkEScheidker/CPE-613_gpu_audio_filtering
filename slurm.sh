#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu 1000mb
#SBATCH --ntasks 1
nvcc -o main convol.cu
./main
