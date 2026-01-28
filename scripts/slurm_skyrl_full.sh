#!/bin/bash
#SBATCH --job-name=skyrl_full
#SBATCH --output=logs/skyrl_full_%j.out
#SBATCH --error=logs/skyrl_full_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=main

# SkyRL GRPO Full Training
#
# This script runs full GRPO training on multi-task reasoning datasets
# using SkyRL for faster single-GPU training.
#
# Expected duration: 6-10 hours depending on dataset size
#
# Usage:
#   sbatch scripts/slurm_skyrl_full.sh
#
# With custom settings:
#   sbatch --export=MODEL=Qwen/Qwen2.5-14B-Instruct,MATH=15000 scripts/slurm_skyrl_full.sh

set -e

# Create logs directory
mkdir -p logs

# Activate environment
source ~/.bashrc

if command -v conda &> /dev/null; then
    conda activate lore 2>/dev/null || echo "No conda env 'lore', using base"
fi

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Print environment info
echo "=============================================="
echo "SkyRL GRPO Full Training"
echo "=============================================="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "SLURM Job ID: ${SLURM_JOB_ID:-local}"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo ""

# Check GPU
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || echo "No GPU"
echo ""

# Set environment variables
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-~/.cache/huggingface}

# Configuration (can be overridden)
MODEL=${MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
MATH=${MATH:-10000}
LOGIC=${LOGIC:-5000}
CODE=${CODE:-2000}
GROUP_SIZE=${GROUP_SIZE:-4}
BATCH_SIZE=${BATCH_SIZE:-8}
OUTPUT="results/skyrl_full_${SLURM_JOB_ID:-local}"

echo "Configuration:"
echo "  Model: ${MODEL}"
echo "  Math samples: ${MATH}"
echo "  Logic samples: ${LOGIC}"
echo "  Code samples: ${CODE}"
echo "  Group size: ${GROUP_SIZE}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Output: ${OUTPUT}"
echo "=============================================="
echo ""

# Run training
python scripts/train_skyrl.py \
    --model "${MODEL}" \
    --output "${OUTPUT}" \
    --math-samples "${MATH}" \
    --logic-samples "${LOGIC}" \
    --code-samples "${CODE}" \
    --group-size "${GROUP_SIZE}" \
    --batch-size "${BATCH_SIZE}" \
    --use-vllm \
    --wandb \
    --seed 42

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "Training COMPLETE!"
    echo "Model saved to: ${OUTPUT}"
    echo "=============================================="
else
    echo ""
    echo "=============================================="
    echo "Training FAILED with exit code $EXIT_CODE"
    echo "=============================================="
fi

exit $EXIT_CODE
