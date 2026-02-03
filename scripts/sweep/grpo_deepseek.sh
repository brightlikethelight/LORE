#!/bin/bash
#SBATCH --job-name=grpo_deepseek
#SBATCH --output=/home/mila/t/thomas.jiralerspong/LORE/logs/grpo_deepseek_%j.out
#SBATCH --error=/home/mila/t/thomas.jiralerspong/LORE/logs/grpo_deepseek_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=long

# GRPO Sweep: deepseek
# lr=1e-06, beta=0.001, num_gen=16, temp=1.0

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
echo "GRPO Sweep: deepseek"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Config: lr=1e-06, beta=0.001, num_gen=16, temp=1.0"
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
OUTPUT_DIR="$PROJECT_DIR/results/sweep_deepseek_20260122_121415"

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

# Apply sweep hyperparameters
config.grpo.learning_rate = 1e-06
config.grpo.kl_coef = 0.001
config.grpo.num_generations = 16
config.grpo.generation_batch_size = 16
config.grpo.temperature = 1.0
config.grpo.output_dir = '$OUTPUT_DIR'
config.grpo.logging_steps = 10
config.grpo.save_steps = 50

# Enable wandb
config.wandb.enabled = True
config.wandb.project = 'lore-grpo-sweep'
config.wandb.tags = ['sweep', 'deepseek', 'lr=1e-06', 'beta=0.001']

# Dataset
config.dataset.train_size = $TRAIN_SAMPLES
config.dataset.val_size = $VAL_SAMPLES

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

print(f'Training complete: {{result.training_steps}} steps, loss={{result.final_loss:.4f}}')

# Print improvement summary
if 'improvement' in result.metrics:
    print('\nAccuracy Improvement:')
    for task, delta in result.metrics['improvement'].items():
        baseline = result.metrics.get('baseline_accuracy', {}).get(task, 0)
        final = baseline + delta
        sign = '+' if delta >= 0 else ''
        print(f'  {{task}}: {{baseline:.1%}} -> {{final:.1%}} ({{sign}}{{delta:.1%}})')
"

echo ""
echo "======================================"
echo "Completed at $(date)"
echo "Results: $OUTPUT_DIR"
echo "======================================"
