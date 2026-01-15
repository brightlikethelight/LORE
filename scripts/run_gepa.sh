#!/bin/bash
# Run GEPA (Genetic Evolution of Prompt Agents) optimization
#
# Usage: ./scripts/run_gepa.sh [options]
#
# Options:
#   --dataset       Dataset to optimize on (default: gsm8k)
#   --generations   Number of generations (default: 100)
#   --population    Population size (default: 50)
#   --model         Target model (default: claude-sonnet-4)
#   --output        Output directory (default: results/gepa)
#   --seed          Random seed (default: 42)
#   --wandb         Enable wandb logging (default: false)

set -euo pipefail

# Default values
DATASET="gsm8k"
GENERATIONS=100
POPULATION=50
MODEL="claude-sonnet-4"
OUTPUT_DIR="results/gepa"
SEED=42
WANDB_ENABLED=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --generations)
            GENERATIONS="$2"
            shift 2
            ;;
        --population)
            POPULATION="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --wandb)
            WANDB_ENABLED=true
            shift
            ;;
        -h|--help)
            head -20 "$0" | tail -18
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Verify environment
if [[ -z "${ANTHROPIC_API_KEY:-}" ]] && [[ "$MODEL" == claude* ]]; then
    echo "Error: ANTHROPIC_API_KEY not set"
    exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" ]] && [[ "$MODEL" == gpt* ]]; then
    echo "Error: OPENAI_API_KEY not set"
    exit 1
fi

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUTPUT_DIR}/${DATASET}_${MODEL}_${TIMESTAMP}"
mkdir -p "$RUN_DIR"

echo "=============================================="
echo "GEPA Optimization Run"
echo "=============================================="
echo "Dataset:      $DATASET"
echo "Generations:  $GENERATIONS"
echo "Population:   $POPULATION"
echo "Model:        $MODEL"
echo "Output:       $RUN_DIR"
echo "Seed:         $SEED"
echo "WandB:        $WANDB_ENABLED"
echo "=============================================="

# Log configuration
cat > "$RUN_DIR/run_config.yaml" << EOF
run_type: gepa
timestamp: $TIMESTAMP
dataset: $DATASET
generations: $GENERATIONS
population: $POPULATION
model: $MODEL
seed: $SEED
wandb_enabled: $WANDB_ENABLED
EOF

# Run GEPA optimization
python -m src.cli gepa \
    --dataset "$DATASET" \
    --generations "$GENERATIONS" \
    --population "$POPULATION" \
    --model "$MODEL" \
    --output "$RUN_DIR" \
    --seed "$SEED" \
    $([ "$WANDB_ENABLED" = true ] && echo "--wandb") \
    2>&1 | tee "$RUN_DIR/run.log"

# Check exit status
if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
    echo ""
    echo "Optimization completed successfully!"
    echo "Results saved to: $RUN_DIR"
    echo ""
    echo "Best prompt saved to: $RUN_DIR/best_prompt.txt"
else
    echo ""
    echo "Optimization failed. Check logs at: $RUN_DIR/run.log"
    exit 1
fi
