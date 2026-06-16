#!/bin/bash
#SBATCH --job-name=q_sampling_base  # Name of your job
#SBATCH --account=bccu-dtai-gh      # YOUR EXPLICIT DELTA-AI ACCOUNT FROM QUOTA LOG
#SBATCH --partition=cpu             # Partition type
#SBATCH --nodes=1                   # Request 1 compute node
#SBATCH --ntasks-per-node=1         # Run a single program instance
#SBATCH --cpus-per-task=16          # Request 16 parallel CPU cores on that node
#SBATCH --mem=16g                   # Request 16 Gigabytes of RAM
#SBATCH --time=00:05:00             # Time limit (5 minutes max)
#SBATCH --output=simulation_%j.log  # Log file name matching the Slurm JobID

# Purge default cluster configurations and load python
module purge
module load python/3.10             # Adjust based on 'module avail python' if needed

# Run your parallel script, passing the allocated CPU count as an argument
python classical_env.py $SLURM_CPUS_PER_TASK