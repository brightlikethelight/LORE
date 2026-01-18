# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LORE (Learned Obfuscated Reasoning) is a research framework for studying whether LLMs can learn to obfuscate their reasoning through prompt optimization. The core hypothesis is that prompt-evolved reasoning guidelines trained across diverse tasks will transfer to safety-critical domains and degrade human/AI monitoring effectiveness.

**Two Training Regimes:**
- **Regime A:** Optimizes for task accuracy only
- **Regime B:** Optimizes for accuracy while evading monitor detection (fitness = accuracy × (1 - monitor_suspicion))

## Common Commands

```bash
# Install (dev mode with development dependencies)
pip install -e ".[dev]"

# Install with sandbox support
pip install -e ".[sandbox]"

# Run GEPA prompt optimization
lore gepa --dataset gsm8k --generations 100 --population 50 --regime A

# Evaluate a prompt with inspect-ai
lore evaluate <prompt_path> --task math_reasoning --model gpt-4o

# Run transfer evaluation across models
lore transfer <prompt_path> --models "gpt-4o,gemini-2.0-flash,claude-sonnet-4"

# Run obfuscation monitor on a response
lore monitor <response_path> <prompt_path>

# Testing
pytest tests/                    # Run all tests
pytest tests/test_foo.py -k bar  # Run specific test
pytest --cov=src tests/          # With coverage

# Linting and type checking
ruff check src/
mypy src/

# Docker deployment (vLLM + Redis)
docker compose up -d vllm redis
```

## Architecture

```
src/
├── cli.py                    # Typer CLI entry point (lore command)
├── data/
│   ├── loaders.py            # DataLoader, DataPoint classes
│   └── gym_builder.py        # Multi-task gym with hidden goal injection
├── models/
│   ├── frontier_api.py       # Anthropic, OpenAI, Google clients
│   └── open_weights.py       # vLLM client for open-weight models
├── optimization/
│   ├── base_optimizer.py     # Abstract optimizer interface
│   ├── gepa.py               # GEPA evolutionary optimization
│   ├── promptbreeder.py      # Genetic algorithm optimization
│   ├── mipro.py              # Multi-objective optimization
│   └── grg_mutations.py      # GRG mutation strategies
└── monitors/
    ├── lindner_autorater.py  # CoT obfuscation detection (Lindner et al.)
    ├── ensemble.py           # Multi-judge ensemble monitor
    └── detection.py          # General detection methods

evals/                        # inspect_ai evaluation tasks
configs/                      # YAML configs (datasets, models, gepa)
scripts/                      # Bash scripts for experiments
docker/                       # Container definitions
```

## Key Patterns

**Model Clients:** All model interactions go through unified clients in `src/models/`. Frontier clients handle rate limiting and retries. Use `get_client("openai", model_id="gpt-4o")` to get a client.

**Optimizers:** All inherit from `BaseOptimizer` with async `optimize(trainset, valset)` method. Results are `OptimizationResult` objects with `save()`/`load()` methods.

**Monitors:** `LindnerAutorater` is the primary obfuscation detector. Returns risk levels (LOW/MEDIUM/HIGH) and pattern flags.

**Hidden Goal Injection:** `GymBuilder` injects hidden objectives into 20% of training samples. Three categories: benign (structural), capability_elicitation, misaligned_research.

## Configuration

Copy `.env.example` to `.env` and set API keys:
- `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY` for frontier models
- `WANDB_API_KEY` for experiment tracking
- `HF_TOKEN` for gated HuggingFace datasets
- `VLLM_BASE_URL` for open-weight model serving

Config files in `configs/`:
- `datasets.yaml` - dataset paths and hidden goal templates
- `models.yaml` - model specs and rate limits
- `gepa_config.yaml` - optimization hyperparameters
