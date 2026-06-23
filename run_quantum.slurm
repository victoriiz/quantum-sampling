#!/bin/bash
#SBATCH --job-name=q_ansatz_test  
#SBATCH --account=bccu-dtai-gh      
#SBATCH --partition=ghx4            
#SBATCH --nodes=1                   
#SBATCH --cpus-per-task=8          
#SBATCH --gpus=1                    
#SBATCH --mem=16g                   
#SBATCH --time=00:05:00             
#SBATCH --output=outputs/quantum_test_%j.log  

module purge
module load default
module load python/3.11.9             

# Navigate to your repository script location
cd /work/nvme/bccu/vz8/quantum-sampling

# Execute using your safe home-hosted python environment binary
~/qenv/bin/python quantum_ansatz.py