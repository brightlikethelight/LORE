#!/bin/bash
#SBATCH --job-name=skyrl_smoke
#SBATCH --output=logs/skyrl_smoke_%j.out
#SBATCH --error=logs/skyrl_smoke_%j.err
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=main

# SkyRL GRPO Smoke Test
#
# This script runs a quick smoke test to verify that SkyRL training
# works correctly before committing to a full training run.
#
# Expected duration: ~30-45 minutes (much faster than Verl's 4+ hours)
#
# Usage:
#   sbatch scripts/slurm_skyrl_smoke.sh
#
# Or with custom model:
#   sbatch --export=MODEL=Qwen/Qwen2.5-14B-Instruct scripts/slurm_skyrl_smoke.sh

set -e

# Create logs directory
mkdir -p logs

# Activate environment (adjust as needed for your setup)
source ~/.bashrc

# If using conda
if command -v conda &> /dev/null; then
    conda activate lore 2>/dev/null || echo "No conda env 'lore', using base"
fi

# If using virtualenv
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Print environment info
echo "=============================================="
echo "SkyRL GRPO Smoke Test"
echo "=============================================="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "SLURM Job ID: ${SLURM_JOB_ID:-local}"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Check GPU availability
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || echo "No GPU detected"
echo ""

# Set environment variables
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-~/.cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-~/.cache/huggingface/transformers}

# Default model (can be overridden via --export=MODEL=...)
MODEL=${MODEL:-"Qwen/Qwen2.5-7B-Instruct"}

echo "Model: ${MODEL}"
echo "=============================================="
echo ""

# Run smoke test
python scripts/train_skyrl.py \
    --model "${MODEL}" \
    --smoke-test \
    --output results/skyrl_smoke_${SLURM_JOB_ID:-local} \
    --seed 42

# Check exit status
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "Smoke test PASSED!"
    echo "=============================================="
else
    echo ""
    echo "=============================================="
    echo "Smoke test FAILED with exit code $EXIT_CODE"
    echo "=============================================="
fi

exit $EXIT_CODE
