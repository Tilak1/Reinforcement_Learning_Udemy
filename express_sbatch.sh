#!/bin/bash
#SBATCH --job-name=express_job      # Job name
#SBATCH --partition=express         # Partition name
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks-per-node=1        # Number of tasks per node
#SBATCH --mem=10G                  # Memory per node (example: 100GB)
#SBATCH --time=00:30:00             # Wall time (30 minutes)
#SBATCH --output=job_output_%j.txt  # Output file name with Job ID
#SBATCH --error=job_error_%j.txt    # Error file name with Job ID

# Load necessary modules (example)
#module load python/3.8.5
#module load anaconda3/2022.05
#module load cuda/11.8
# Activate the conda environment

# Load required modules
module load anaconda3/2022.05 cuda/12.1 gcc/11.1.0 

# Activate the Conda environment
source activate pytorch_env

# Execute your application
python Lunar_udemy_indstructor_random.py
