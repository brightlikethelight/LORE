#!/bin/bash
# Launch parallel optimization experiments
#
# Usage: ./scripts/launch_parallel.sh [options]
#
# Options:
#   --config        Path to experiment config YAML (required)
#   --max-jobs      Maximum parallel jobs (default: 4)
#   --output        Output base directory (default: results/parallel)
#   --dry-run       Print commands without executing

set -euo pipefail

# Default values
CONFIG_PATH=""
MAX_JOBS=4
OUTPUT_BASE="results/parallel"
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --max-jobs)
            MAX_JOBS="$2"
            shift 2
            ;;
        --output)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            head -16 "$0" | tail -14
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate config
if [[ -z "$CONFIG_PATH" ]]; then
    echo "Error: --config is required"
    exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Error: Config file not found: $CONFIG_PATH"
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUTPUT_BASE}/batch_${TIMESTAMP}"

echo "=============================================="
echo "Parallel Experiment Launch"
echo "=============================================="
echo "Config:       $CONFIG_PATH"
echo "Max Jobs:     $MAX_JOBS"
echo "Output:       $RUN_DIR"
echo "Dry Run:      $DRY_RUN"
echo "=============================================="

# Create output directory
mkdir -p "$RUN_DIR"
cp "$CONFIG_PATH" "$RUN_DIR/experiment_config.yaml"

# Parse config and generate jobs
python << PYTHON
import yaml
from pathlib import Path

config_path = "$CONFIG_PATH"
run_dir = "$RUN_DIR"
dry_run = $([[ "$DRY_RUN" == "true" ]] && echo "True" || echo "False")

with open(config_path) as f:
    config = yaml.safe_load(f)

experiments = config.get("experiments", [])
commands = []

for i, exp in enumerate(experiments):
    exp_name = exp.get("name", f"exp_{i}")
    optimizer = exp.get("optimizer", "gepa")
    dataset = exp.get("dataset", "gsm8k")
    model = exp.get("model", "claude-sonnet-4")
    generations = exp.get("generations", 50)
    population = exp.get("population", 30)
    seed = exp.get("seed", 42 + i)

    output_dir = f"{run_dir}/{exp_name}"

    cmd = f"""./scripts/run_gepa.sh \\
    --dataset {dataset} \\
    --model {model} \\
    --generations {generations} \\
    --population {population} \\
    --seed {seed} \\
    --output {output_dir} \\
    --wandb"""

    commands.append((exp_name, cmd))

# Write commands to file
with open(f"{run_dir}/commands.txt", "w") as f:
    for name, cmd in commands:
        f.write(f"# {name}\n{cmd}\n\n")

# Print summary
print(f"\nGenerated {len(commands)} experiment commands")
print(f"Commands saved to: {run_dir}/commands.txt")

if dry_run:
    print("\n[DRY RUN] Commands that would be executed:")
    for name, cmd in commands:
        print(f"\n--- {name} ---")
        print(cmd)
else:
    # Write job script for GNU Parallel
    job_file = f"{run_dir}/jobs.txt"
    with open(job_file, "w") as f:
        for name, cmd in commands:
            # Convert multi-line to single line
            single_line = cmd.replace("\\\n", " ").replace("  ", " ")
            f.write(f"{single_line}\n")

    print(f"Job file written to: {job_file}")
PYTHON

# Execute if not dry run
if [[ "$DRY_RUN" == "false" ]]; then
    echo ""
    echo "Starting parallel execution..."
    echo ""

    # Check for GNU Parallel
    if command -v parallel &> /dev/null; then
        parallel --jobs "$MAX_JOBS" --progress --joblog "$RUN_DIR/parallel.log" < "$RUN_DIR/jobs.txt"
    else
        echo "GNU Parallel not found, falling back to xargs..."
        xargs -P "$MAX_JOBS" -I {} bash -c '{}' < "$RUN_DIR/jobs.txt"
    fi

    echo ""
    echo "=============================================="
    echo "All experiments completed!"
    echo "Results in: $RUN_DIR"
    echo "=============================================="
fi
