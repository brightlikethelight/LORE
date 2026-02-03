#!/bin/bash
#SBATCH --job-name=grpo_qwen3b
#SBATCH --output=/home/mila/t/thomas.jiralerspong/LORE/logs/grpo_qwen3b_%j.out
#SBATCH --error=/home/mila/t/thomas.jiralerspong/LORE/logs/grpo_qwen3b_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=long

# GRPO Sweep: Qwen2.5-3B-Instruct (medium model)
# model=Qwen2.5-3B-Instruct, lr=3e-06, beta=0.001, num_gen=8, temp=1.0

set -e

PROJECT_DIR=/home/mila/t/thomas.jiralerspong/LORE
cd $PROJECT_DIR

export HF_HOME=$SCRATCH/hugging_face_cache
export TRANSFORMERS_CACHE=$SCRATCH/hugging_face_cache
export HF_DATASETS_CACHE=$SCRATCH/hugging_face_cache/datasets
export PIP_CACHE_DIR=$SCRATCH/cache/pip
export UV_CACHE_DIR=$SCRATCH/uv_cache
export WANDB_PROJECT="lore-grpo-sweep"
export WANDB_DIR=$SCRATCH/wandb

mkdir -p logs $HF_HOME $PIP_CACHE_DIR $WANDB_DIR

echo "======================================"
echo "GRPO Sweep: Qwen2.5-3B-Instruct"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Config: model=Qwen2.5-3B, lr=3e-06, beta=0.001, num_gen=8, temp=1.0"
echo "Start time: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "GPU: N/A"
echo ""

# Setup environment
module load python/3.10
python -m venv $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate

pip install --upgrade pip wheel setuptools --quiet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir --quiet
pip install -e ".[rl]" --quiet
pip install wandb --quiet

# Training
TRAIN_SAMPLES=2000
VAL_SAMPLES=200
MAX_STEPS=200
OUTPUT_DIR="$PROJECT_DIR/results/sweep_qwen3b_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "Training: $TRAIN_SAMPLES samples, $MAX_STEPS steps"
echo "Output: $OUTPUT_DIR"
echo ""

python -c "
import sys
sys.path.insert(0, '.')

from src.rl.config import RLConfig, load_config
from src.rl.grpo_trainer import LOREGRPOTrainer
from src.rl.data_pipeline import load_ultrainteract_for_grpo
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(name)s: %(message)s')

# Load and modify config
config = load_config()

# Use medium model
config.grpo.model_name = 'Qwen/Qwen2.5-3B-Instruct'

# Apply sweep hyperparameters (best from 7B runs)
config.grpo.learning_rate = 3e-06
config.grpo.kl_coef = 0.001
config.grpo.num_generations = 8
config.grpo.temperature = 1.0
config.grpo.output_dir = '$OUTPUT_DIR'
config.grpo.logging_steps = 10
config.grpo.save_steps = 50

# Enable wandb
config.wandb.enabled = True
config.wandb.project = 'lore-grpo-sweep'
config.wandb.tags = ['sweep', 'qwen3b', 'medium_model', 'lr=3e-06', 'beta=0.001']

# Dataset
config.dataset.train_size = $TRAIN_SAMPLES
config.dataset.val_size = $VAL_SAMPLES

print(f'Model: {config.grpo.model_name}')
print(f'Config: lr={config.grpo.learning_rate}, beta={config.grpo.kl_coef}, num_gen={config.grpo.num_generations}, temp={config.grpo.temperature}')

# Create trainer
trainer = LOREGRPOTrainer(config=config)

# Load data
datasets = load_ultrainteract_for_grpo(
    train_size=$TRAIN_SAMPLES,
    val_size=$VAL_SAMPLES,
    tokenizer=trainer.tokenizer,
    seed=config.dataset.seed,
    task_types=config.dataset.task_types,
)

# Train with baseline evaluation
result = trainer.train(
    train_dataset=datasets['train'],
    eval_dataset=datasets['val'],
    max_steps=$MAX_STEPS,
    evaluate_baseline=True,
    baseline_samples=100,
)

print(f'Training complete: {result.training_steps} steps, loss={result.final_loss:.4f}')

# Print improvement summary
if 'improvement' in result.metrics:
    print('\nAccuracy Improvement:')
    for task, delta in result.metrics['improvement'].items():
        baseline = result.metrics.get('baseline_accuracy', {}).get(task, 0)
        final = baseline + delta
        sign = '+' if delta >= 0 else ''
        print(f'  {task}: {baseline:.1%} -> {final:.1%} ({sign}{delta:.1%})')
"

echo ""
echo "======================================"
echo "Completed at $(date)"
echo "Results: $OUTPUT_DIR"
echo "======================================"
