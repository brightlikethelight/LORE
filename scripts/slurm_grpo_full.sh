#!/bin/bash
#SBATCH --job-name=grpo_full
#SBATCH --output=/home/mila/t/thomas.jiralerspong/LORE/logs/grpo_full_%j.out
#SBATCH --error=/home/mila/t/thomas.jiralerspong/LORE/logs/grpo_full_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=long

# Full GRPO Training on UltraInteract
# Trains Qwen2.5-7B with LoRA on 5000 samples

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
echo "GRPO Full Training"
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

# Training configuration
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-5000}"
VAL_SAMPLES="${VAL_SAMPLES:-500}"
OUTPUT_DIR="$PROJECT_DIR/results/grpo_full_$(date +%Y%m%d_%H%M%S)"
WANDB_FLAG="${WANDB_ENABLED:+--wandb}"

echo "Training Configuration:"
echo "  Model: $MODEL"
echo "  Train samples: $TRAIN_SAMPLES"
echo "  Val samples: $VAL_SAMPLES"
echo "  Output: $OUTPUT_DIR"
echo "  WandB: ${WANDB_ENABLED:-disabled}"
echo ""

python -m src.cli rl-train \
    --model "$MODEL" \
    --output "$OUTPUT_DIR" \
    --samples $TRAIN_SAMPLES \
    --val-samples $VAL_SAMPLES \
    $WANDB_FLAG

echo ""
echo "======================================"
echo "Training completed at $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "======================================"
