#!/bin/bash
#SBATCH --job-name=gepa_v6_notrunc
#SBATCH --output=/home/mila/t/thomas.jiralerspong/LORE/logs/slurm_v6_%j.out
#SBATCH --error=/home/mila/t/thomas.jiralerspong/LORE/logs/slurm_v6_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=long-cpu

# GEPA Evolution v6 - Fixed truncation in reflection feedback

cd /home/mila/t/thomas.jiralerspong/LORE

# Activate environment
source .venv/bin/activate

# Set cache directories
export HF_HOME=/home/mila/t/thomas.jiralerspong/.cache/huggingface
export TRANSFORMERS_CACHE=/home/mila/t/thomas.jiralerspong/.cache/huggingface

# Run evolution
echo "Starting GEPA Evolution v6 (no truncation) at $(date)"
python scripts/run_evolution_v6_notrunc.py
echo "Finished at $(date)"
