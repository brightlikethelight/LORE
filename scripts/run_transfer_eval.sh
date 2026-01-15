#!/bin/bash
# Run transfer evaluation of optimized prompts across models
#
# Usage: ./scripts/run_transfer_eval.sh [options]
#
# Options:
#   --prompt        Path to optimized prompt file (required)
#   --task          Evaluation task (default: math_reasoning)
#   --models        Comma-separated list of models to evaluate
#   --samples       Number of samples per model (default: 200)
#   --output        Output directory (default: results/transfer)
#   --parallel      Number of parallel evaluations (default: 3)

set -euo pipefail

# Default values
PROMPT_PATH=""
TASK="math_reasoning"
MODELS="gpt-4o,gemini-2.0-flash,claude-sonnet-4"
SAMPLES=200
OUTPUT_DIR="results/transfer"
PARALLEL=3

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --prompt)
            PROMPT_PATH="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --models)
            MODELS="$2"
            shift 2
            ;;
        --samples)
            SAMPLES="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        -h|--help)
            head -18 "$0" | tail -16
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate prompt path
if [[ -z "$PROMPT_PATH" ]]; then
    echo "Error: --prompt is required"
    exit 1
fi

if [[ ! -f "$PROMPT_PATH" ]]; then
    echo "Error: Prompt file not found: $PROMPT_PATH"
    exit 1
fi

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUTPUT_DIR}/transfer_${TASK}_${TIMESTAMP}"
mkdir -p "$RUN_DIR"

echo "=============================================="
echo "Transfer Evaluation Run"
echo "=============================================="
echo "Prompt:       $PROMPT_PATH"
echo "Task:         $TASK"
echo "Models:       $MODELS"
echo "Samples:      $SAMPLES"
echo "Output:       $RUN_DIR"
echo "Parallel:     $PARALLEL"
echo "=============================================="

# Copy prompt for reference
cp "$PROMPT_PATH" "$RUN_DIR/optimized_prompt.txt"

# Save config
cat > "$RUN_DIR/eval_config.yaml" << EOF
run_type: transfer_eval
timestamp: $TIMESTAMP
prompt_path: $PROMPT_PATH
task: $TASK
models: $(echo "$MODELS" | tr ',' '\n' | sed 's/^/  - /')
samples: $SAMPLES
parallel: $PARALLEL
EOF

# Read the optimized prompt
OPTIMIZED_PROMPT=$(cat "$PROMPT_PATH")

# Convert comma-separated models to array
IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"

# Function to run evaluation for a single model
run_single_eval() {
    local model=$1
    local model_safe=$(echo "$model" | tr '/' '_')

    echo "[$(date +%H:%M:%S)] Starting evaluation for $model..."

    # Run inspect_ai evaluation
    inspect eval evals/tasks/${TASK}.py \
        --model "$model" \
        --limit "$SAMPLES" \
        --log-dir "$RUN_DIR/logs/$model_safe" \
        -T "optimized_prompt=$OPTIMIZED_PROMPT" \
        2>&1 | tee "$RUN_DIR/logs/${model_safe}.log"

    echo "[$(date +%H:%M:%S)] Completed evaluation for $model"
}

export -f run_single_eval
export OPTIMIZED_PROMPT TASK SAMPLES RUN_DIR

# Run evaluations in parallel
mkdir -p "$RUN_DIR/logs"

echo ""
echo "Running evaluations..."
echo ""

printf '%s\n' "${MODEL_ARRAY[@]}" | xargs -P "$PARALLEL" -I {} bash -c 'run_single_eval "$@"' _ {}

# Aggregate results
echo ""
echo "=============================================="
echo "Aggregating Results"
echo "=============================================="

python << PYTHON
import json
from pathlib import Path
import yaml

results_dir = Path("$RUN_DIR/logs")
summary = {"models": {}}

for model_dir in results_dir.iterdir():
    if model_dir.is_dir():
        # Find latest log file
        log_files = list(model_dir.glob("*.json"))
        if log_files:
            latest = max(log_files, key=lambda x: x.stat().st_mtime)
            with open(latest) as f:
                data = json.load(f)

            model_name = model_dir.name.replace("_", "/")
            summary["models"][model_name] = {
                "accuracy": data.get("results", {}).get("accuracy", {}).get("value", "N/A"),
                "samples": data.get("eval", {}).get("samples", "N/A"),
            }

# Write summary
summary_path = Path("$RUN_DIR/summary.yaml")
with open(summary_path, "w") as f:
    yaml.dump(summary, f, default_flow_style=False)

print("Results Summary:")
print("-" * 40)
for model, metrics in summary["models"].items():
    print(f"{model}: {metrics['accuracy']}")
PYTHON

echo ""
echo "Transfer evaluation complete!"
echo "Results saved to: $RUN_DIR"
echo "Summary: $RUN_DIR/summary.yaml"
