#!/bin/bash
#SBATCH --job-name=grpo_train
#SBATCH --output=/home/mila/t/thomas.jiralerspong/LORE/logs/grpo_%j.out
#SBATCH --error=/home/mila/t/thomas.jiralerspong/LORE/logs/grpo_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=main

# GRPO Training on UltraInteract
# Uses TRL's GRPO with LoRA for memory-efficient 7B model training

set -e

cd /home/mila/t/thomas.jiralerspong/LORE

# Activate environment
source .venv/bin/activate

# Set cache directories
export HF_HOME=/home/mila/t/thomas.jiralerspong/.cache/huggingface
export TRANSFORMERS_CACHE=/home/mila/t/thomas.jiralerspong/.cache/huggingface
export HF_DATASETS_CACHE=/home/mila/t/thomas.jiralerspong/.cache/huggingface/datasets

# Ensure logs directory exists
mkdir -p logs

echo "======================================"
echo "GRPO Training on UltraInteract"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# Install RL dependencies if not present
if ! python -c "import trl" 2>/dev/null; then
    echo "Installing RL dependencies..."
    python -m pip install -e ".[rl]" --quiet
    echo "RL dependencies installed."
fi

# Verify installation
echo "Checking dependencies..."
python -c "import trl; import peft; import accelerate; print('TRL:', trl.__version__); print('PEFT:', peft.__version__)"
echo ""

# Training parameters (can be overridden via environment variables)
MODEL=${MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-5000}
VAL_SAMPLES=${VAL_SAMPLES:-500}
MAX_STEPS=${MAX_STEPS:-""}
OUTPUT_DIR=${OUTPUT_DIR:-"results/grpo_$(date +%Y%m%d_%H%M%S)"}
WANDB_ENABLED=${WANDB_ENABLED:-"false"}

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Train samples: $TRAIN_SAMPLES"
echo "  Val samples: $VAL_SAMPLES"
echo "  Max steps: ${MAX_STEPS:-'full epochs'}"
echo "  Output: $OUTPUT_DIR"
echo "  WandB: $WANDB_ENABLED"
echo ""

# Build command
CMD="python -m src.cli rl-train"
CMD="$CMD --model $MODEL"
CMD="$CMD --output $OUTPUT_DIR"
CMD="$CMD --samples $TRAIN_SAMPLES"
CMD="$CMD --val-samples $VAL_SAMPLES"

if [ -n "$MAX_STEPS" ]; then
    CMD="$CMD --max-steps $MAX_STEPS"
fi

if [ "$WANDB_ENABLED" = "true" ]; then
    CMD="$CMD --wandb"
fi

echo "Running: $CMD"
echo "======================================"
echo ""

# Run training
$CMD

echo ""
echo "======================================"
echo "Training completed at $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "======================================"
