#!/usr/bin/env python3
"""GRPO Hyperparameter Sweep Launcher.

Launches multiple SLURM jobs with different hyperparameter configurations.
Based on best practices from DeepSeek R1 and recent GRPO research.

Usage:
    python scripts/grpo_sweep.py --dry-run  # Preview jobs
    python scripts/grpo_sweep.py            # Launch jobs
"""

import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from itertools import product

# Sweep configurations based on research
SWEEP_CONFIGS = [
    # Config 1: Conservative (stable)
    {"name": "conservative", "lr": 5e-7, "beta": 0.001, "num_gen": 8, "temp": 1.0},

    # Config 2: DeepSeek-inspired
    {"name": "deepseek", "lr": 1e-6, "beta": 0.001, "num_gen": 16, "temp": 1.0},

    # Config 3: Aggressive learning rate
    {"name": "high_lr", "lr": 3e-6, "beta": 0.001, "num_gen": 8, "temp": 1.0},

    # Config 4: No KL penalty (recent findings suggest this works)
    {"name": "no_kl", "lr": 1e-6, "beta": 0.0, "num_gen": 8, "temp": 1.0},

    # Config 5: Higher KL penalty
    {"name": "high_kl", "lr": 1e-6, "beta": 0.01, "num_gen": 8, "temp": 1.0},

    # Config 6: More generations (better advantage estimation)
    {"name": "many_gen", "lr": 1e-6, "beta": 0.001, "num_gen": 32, "temp": 1.0},

    # Config 7: Lower temperature (less exploration)
    {"name": "low_temp", "lr": 1e-6, "beta": 0.001, "num_gen": 8, "temp": 0.7},
]

SLURM_TEMPLATE = '''#!/bin/bash
#SBATCH --job-name=grpo_{name}
#SBATCH --output=/home/mila/t/thomas.jiralerspong/LORE/logs/grpo_{name}_%j.out
#SBATCH --error=/home/mila/t/thomas.jiralerspong/LORE/logs/grpo_{name}_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=long

# GRPO Sweep: {name}
# lr={lr}, beta={beta}, num_gen={num_gen}, temp={temp}

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
echo "GRPO Sweep: {name}"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Config: lr={lr}, beta={beta}, num_gen={num_gen}, temp={temp}"
echo "Start time: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "GPU: N/A"
echo ""

# Setup environment
module load python/3.10
python -m venv $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate

pip install --upgrade pip wheel setuptools --quiet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet
pip install -e ".[rl]" --quiet
pip install wandb --quiet

# Training
TRAIN_SAMPLES=2000
VAL_SAMPLES=200
MAX_STEPS=200
OUTPUT_DIR="$PROJECT_DIR/results/sweep_{name}_{timestamp}"

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
config.grpo.learning_rate = {lr}
config.grpo.kl_coef = {beta}
config.grpo.num_generations = {num_gen}
config.grpo.temperature = {temp}
config.grpo.output_dir = '$OUTPUT_DIR'
config.grpo.logging_steps = 10
config.grpo.save_steps = 50

# Enable wandb
config.wandb.enabled = True
config.wandb.project = 'lore-grpo-sweep'
config.wandb.tags = ['sweep', '{name}', 'lr={lr}', 'beta={beta}']

# Dataset
config.dataset.train_size = $TRAIN_SAMPLES
config.dataset.val_size = $VAL_SAMPLES

print(f'Config: lr={{config.grpo.learning_rate}}, beta={{config.grpo.kl_coef}}, num_gen={{config.grpo.num_generations}}, temp={{config.grpo.temperature}}')

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

print(f'Training complete: {{{{result.training_steps}}}} steps, loss={{{{result.final_loss:.4f}}}}')

# Print improvement summary
if 'improvement' in result.metrics:
    print('\\nAccuracy Improvement:')
    for task, delta in result.metrics['improvement'].items():
        baseline = result.metrics.get('baseline_accuracy', {{}}).get(task, 0)
        final = baseline + delta
        sign = '+' if delta >= 0 else ''
        print(f'  {{{{task}}}}: {{{{baseline:.1%}}}} -> {{{{final:.1%}}}} ({{{{sign}}}}{{{{delta:.1%}}}})')
"

echo ""
echo "======================================"
echo "Completed at $(date)"
echo "Results: $OUTPUT_DIR"
echo "======================================"
'''


def main():
    parser = argparse.ArgumentParser(description="Launch GRPO hyperparameter sweep")
    parser.add_argument("--dry-run", action="store_true", help="Print jobs without launching")
    parser.add_argument("--configs", nargs="+", help="Specific configs to run (by name)")
    args = parser.parse_args()

    project_dir = Path("/home/mila/t/thomas.jiralerspong/LORE")
    scripts_dir = project_dir / "scripts" / "sweep"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    configs = SWEEP_CONFIGS
    if args.configs:
        configs = [c for c in configs if c["name"] in args.configs]

    print(f"GRPO Hyperparameter Sweep")
    print(f"=" * 50)
    print(f"Launching {len(configs)} configurations:")
    print()

    job_ids = []
    for config in configs:
        script_content = SLURM_TEMPLATE.format(
            name=config["name"],
            lr=config["lr"],
            beta=config["beta"],
            num_gen=config["num_gen"],
            temp=config["temp"],
            timestamp=timestamp,
        )

        script_path = scripts_dir / f"grpo_{config['name']}.sh"

        print(f"  {config['name']:15} lr={config['lr']:.0e} beta={config['beta']:.3f} "
              f"num_gen={config['num_gen']:2} temp={config['temp']}")

        if args.dry_run:
            continue

        # Write script
        with open(script_path, "w") as f:
            f.write(script_content)
        script_path.chmod(0o755)

        # Submit job
        result = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            job_ids.append((config["name"], job_id))
            print(f"    -> Submitted job {job_id}")
        else:
            print(f"    -> FAILED: {result.stderr}")

    print()
    if args.dry_run:
        print("Dry run - no jobs submitted")
    else:
        print(f"Submitted {len(job_ids)} jobs:")
        for name, jid in job_ids:
            print(f"  {name}: {jid}")
        print()
        print("Monitor with: squeue -u $USER | grep grpo")
        print("WandB project: lore-grpo-sweep")


if __name__ == "__main__":
    main()
