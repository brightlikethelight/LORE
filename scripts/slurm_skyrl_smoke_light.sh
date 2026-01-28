#!/bin/bash
#SBATCH --job-name=skyrl_smoke
#SBATCH --output=logs/skyrl_smoke_%j.out
#SBATCH --error=logs/skyrl_smoke_%j.err
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=main

# SkyRL GRPO Smoke Test (lightweight version)
# Uses any available GPU type

set -e
mkdir -p logs

# Activate environment
source ~/.bashrc
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "=============================================="
echo "SkyRL GRPO Smoke Test (TRL-based)"
echo "=============================================="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "SLURM Job ID: ${SLURM_JOB_ID:-local}"
echo "Python: $(which python)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || echo "No GPU"
echo ""

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-~/.cache/huggingface}

# Use smaller model for lighter GPU
MODEL=${MODEL:-"Qwen/Qwen2.5-1.5B-Instruct"}

echo "Model: ${MODEL}"
echo "=============================================="

# Use smaller model and shorter responses for faster testing
python scripts/train_skyrl.py \
    --model "${MODEL}" \
    --smoke-test \
    --no-8bit \
    --no-vllm \
    --max-response-length 256 \
    --output results/skyrl_smoke_${SLURM_JOB_ID:-local} \
    --seed 42

echo ""
echo "=============================================="
if [ $? -eq 0 ]; then
    echo "Smoke test PASSED!"
else
    echo "Smoke test FAILED!"
fi
echo "=============================================="
