#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10GB
#SBATCH --time=02:00:00
#SBATCH --job-name=my_dq_learning_job
#SBATCH --output=job_output_%j.txt
#SBATCH --mail-user=marupilla.t@northeastern.edu
#SBATCH --mail-type=ALL


# Load necessary modules
module load anaconda3/2022.05
module load cuda/11.8

# Activate the Conda environment
source activate pytorch_env

# Run the Python script
python DQ_Learning.py
