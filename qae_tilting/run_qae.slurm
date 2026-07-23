#!/bin/bash
#SBATCH --job-name=qae_validate
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=qae_%j.out
#SBATCH --error=qae_%j.err

module load cuda/12.2
module load python/3.11
source ~/qenv/bin/activate

python qae_optimized_gpu.py
