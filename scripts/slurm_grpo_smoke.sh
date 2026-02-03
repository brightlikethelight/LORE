#!/bin/bash
#SBATCH --job-name=grpo_smoke
#SBATCH --output=/home/mila/t/thomas.jiralerspong/LORE/logs/grpo_smoke_%j.out
#SBATCH --error=/home/mila/t/thomas.jiralerspong/LORE/logs/grpo_smoke_%j.err
#SBATCH --time=3:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=main

# GRPO Smoke Test - Quick verification with Verl trainer

set -e

PROJECT_DIR=/home/mila/t/thomas.jiralerspong/LORE
cd $PROJECT_DIR

# Set cache directories
export HF_HOME=/home/mila/t/thomas.jiralerspong/.cache/huggingface
export TRANSFORMERS_CACHE=/home/mila/t/thomas.jiralerspong/.cache/huggingface
export HF_DATASETS_CACHE=/home/mila/t/thomas.jiralerspong/.cache/huggingface/datasets
export PIP_CACHE_DIR=/home/mila/t/thomas.jiralerspong/.cache/pip

mkdir -p logs $HF_HOME $PIP_CACHE_DIR

echo "======================================"
echo "GRPO Smoke Test (Verl Trainer)"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "GPU: N/A"
echo ""

# Activate the project virtualenv
echo "Activating project environment..."
source $PROJECT_DIR/.venv/bin/activate

# Verify GPU
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# Smoke test: 20 samples, 5 steps (reduced for serial generation fallback)
echo "Running smoke test..."
echo "  Math samples: 10"
echo "  Logic samples: 6"
echo "  Code samples: 4"
echo "  Max steps: 5"
echo "  Group size: 2"
echo "  Expected time: ~30 minutes"
echo ""

python scripts/train_verl.py --smoke-test

echo ""
echo "======================================"
echo "Smoke test completed at $(date)"
echo "======================================"
