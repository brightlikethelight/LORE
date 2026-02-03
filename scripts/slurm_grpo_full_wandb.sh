#!/bin/bash
#SBATCH --job-name=grpo_full_wandb
#SBATCH --output=/home/mila/t/thomas.jiralerspong/LORE/logs/grpo_full_wandb_%j.out
#SBATCH --error=/home/mila/t/thomas.jiralerspong/LORE/logs/grpo_full_wandb_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=long

# Full GRPO Training with WandB + Accuracy Tracking
# 5000 samples, 500 steps

set -e

PROJECT_DIR=/home/mila/t/thomas.jiralerspong/LORE
cd $PROJECT_DIR

# Set cache directories
export HF_HOME=/home/mila/t/thomas.jiralerspong/.cache/huggingface
export TRANSFORMERS_CACHE=/home/mila/t/thomas.jiralerspong/.cache/huggingface
export HF_DATASETS_CACHE=/home/mila/t/thomas.jiralerspong/.cache/huggingface/datasets
export PIP_CACHE_DIR=/home/mila/t/thomas.jiralerspong/.cache/pip

# WandB settings
export WANDB_PROJECT="lore-grpo"
export WANDB_ENTITY=""  # Set your entity if needed
export WANDB_DIR="$PROJECT_DIR/wandb"

mkdir -p logs $HF_HOME $PIP_CACHE_DIR $WANDB_DIR

echo "======================================"
echo "GRPO Full Training with WandB"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "GPU: N/A"
echo ""

# Create virtualenv in SLURM_TMPDIR
echo "Setting up environment in SLURM_TMPDIR..."
module load python/3.10
python -m venv $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate

# Install dependencies
echo "Installing base dependencies..."
pip install --upgrade pip wheel setuptools --quiet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet

echo "Installing LORE with RL dependencies..."
pip install -e ".[rl]" --quiet
pip install wandb --quiet

# Verify
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import trl; import peft; print(f'TRL: {trl.__version__}, PEFT: {peft.__version__}')"
python -c "import wandb; print(f'WandB: {wandb.__version__}')"
echo ""

# Training configuration
TRAIN_SAMPLES=5000
VAL_SAMPLES=500
MAX_STEPS=500
OUTPUT_DIR="$PROJECT_DIR/results/grpo_full_wandb_$(date +%Y%m%d_%H%M%S)"

echo "Training Configuration:"
echo "  Model: Qwen/Qwen2.5-7B-Instruct"
echo "  Train samples: $TRAIN_SAMPLES"
echo "  Val samples: $VAL_SAMPLES"
echo "  Max steps: $MAX_STEPS"
echo "  WandB: ENABLED (project: $WANDB_PROJECT)"
echo "  Accuracy tracking: ENABLED"
echo "  Output: $OUTPUT_DIR"
echo ""

# Run training with WandB enabled
python -m src.cli rl-train \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --output "$OUTPUT_DIR" \
    --samples $TRAIN_SAMPLES \
    --val-samples $VAL_SAMPLES \
    --max-steps $MAX_STEPS \
    --wandb

echo ""
echo "======================================"
echo "Training completed at $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Final Accuracy Summary:"
cat "$OUTPUT_DIR/training_result.json" | python -c "
import json, sys
data = json.load(sys.stdin)
print('Accuracy by Task Type:')
counts = data.get('metrics', {}).get('accuracy_counts', {})
for task, c in counts.items():
    if isinstance(c, dict) and c.get('total', 0) > 0:
        acc = c['correct'] / c['total']
        print(f'  {task}: {acc:.1%} ({c[\"correct\"]}/{c[\"total\"]})')
"
echo "======================================"
