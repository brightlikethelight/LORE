# LORE: Learned Obfuscated Reasoning Experiments

A research framework for studying prompt optimization and its implications for AI alignment and safety.

## Installation

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Usage

```bash
# Evolve general reasoning prompt on UltraInteract
lore ultrainteract-evolve --output results/ui_evo --generations 50

# Evaluate monitorability on Caught Red-Handed
lore crh-eval results/ui_evolution/best_prompt.txt --samples 200
```

## License

MIT
