#!/bin/bash
#SBATCH --job-name=grpo_accuracy
#SBATCH --output=/home/mila/t/thomas.jiralerspong/LORE/logs/grpo_accuracy_%j.out
#SBATCH --error=/home/mila/t/thomas.jiralerspong/LORE/logs/grpo_accuracy_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=unkillable

# GRPO Training with Accuracy Tracking
# Tests the new per-task accuracy logging

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
echo "GRPO Training with Accuracy Tracking"
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

# Verify
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import trl; import peft; print(f'TRL: {trl.__version__}, PEFT: {peft.__version__}')"
echo ""

# Run training with 30 steps
TRAIN_SAMPLES=200
VAL_SAMPLES=50
MAX_STEPS=30
OUTPUT_DIR="$PROJECT_DIR/results/grpo_accuracy_$(date +%Y%m%d_%H%M%S)"

echo "Training Configuration:"
echo "  Train samples: $TRAIN_SAMPLES"
echo "  Val samples: $VAL_SAMPLES"
echo "  Max steps: $MAX_STEPS"
echo "  Accuracy tracking: ENABLED"
echo "  Output: $OUTPUT_DIR"
echo ""

python -m src.cli rl-train \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --output "$OUTPUT_DIR" \
    --samples $TRAIN_SAMPLES \
    --val-samples $VAL_SAMPLES \
    --max-steps $MAX_STEPS

echo ""
echo "======================================"
echo "Training completed at $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Checking accuracy metrics..."
cat "$OUTPUT_DIR/training_result.json" | python -c "
import json, sys
data = json.load(sys.stdin)
print('Final Accuracy Metrics:')
for k, v in data.get('metrics', {}).items():
    if 'accuracy' in k:
        if isinstance(v, dict):
            print(f'  {k}:')
            for task, counts in v.items():
                if isinstance(counts, dict):
                    acc = counts['correct'] / counts['total'] if counts['total'] > 0 else 0
                    print(f'    {task}: {acc:.1%} ({counts[\"correct\"]}/{counts[\"total\"]})')
        else:
            print(f'  {k}: {v:.1%}')
"
echo "======================================"
