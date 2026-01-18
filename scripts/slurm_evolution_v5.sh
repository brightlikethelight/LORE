#!/bin/bash
#SBATCH --job-name=gepa_evolution_v5
#SBATCH --output=/home/mila/t/thomas.jiralerspong/LORE/logs/slurm_evolution_v5_%j.out
#SBATCH --error=/home/mila/t/thomas.jiralerspong/LORE/logs/slurm_evolution_v5_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=long-cpu

# Set environment variables
export HF_HOME=/home/mila/t/thomas.jiralerspong/.cache/huggingface

# Activate virtual environment
source /home/mila/t/thomas.jiralerspong/LORE/.venv/bin/activate

# Navigate to project directory
cd /home/mila/t/thomas.jiralerspong/LORE

# Run the evolution script
echo "Starting GEPA Evolution v5 at $(date)"
python scripts/run_evolution_v5_unlimited.py
echo "Finished GEPA Evolution v5 at $(date)"
