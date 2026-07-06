#!/bin/bash
#SBATCH --job-name=q_samp 
#SBATCH --account=bccu-dtai-gh      
#SBATCH --partition=ghx4            # Partition type
#SBATCH --nodes=1                   # Request 1 compute node
#SBATCH --ntasks-per-node=1         # Run a single program instance
#SBATCH --cpus-per-task=16          # Request 16 parallel CPU cores on that node
#SBATCH --gpus=1
#SBATCH --mem=16g                   # Request 16 Gigabytes of RAM
#SBATCH --time=00:05:00             # Time limit (5 minutes max)
#SBATCH --output=simulation_%j.log  # Log file name matching the Slurm JobID

module purge
module load default
module load python/3.11.9          

python scripts/classical_env.py $SLURM_CPUS_PER_TASK