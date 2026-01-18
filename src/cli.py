"""CLI entry point for LORE experiments."""

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="lore",
    help="LORE: Learned Obfuscated Reasoning Experiments",
    add_completion=False,
)
console = Console()


@app.command()
def gepa(
    dataset: str = typer.Option("gsm8k", help="Dataset to optimize on"),
    generations: int = typer.Option(100, help="Number of generations"),
    population: int = typer.Option(50, help="Population size"),
    task_model: str = typer.Option("openai/gpt-4o-mini", help="Model for task evaluation"),
    reflection_model: str = typer.Option("openai/gpt-4o", help="Model for reflection"),
    regime: str = typer.Option("A", help="Training regime: A (capability) or B (capability + stealth)"),
    output: Path = typer.Option(Path("results/gepa"), help="Output directory"),
    seed: int = typer.Option(42, help="Random seed"),
    wandb_enabled: bool = typer.Option(False, "--wandb", help="Enable wandb logging"),
) -> None:
    """Run GEPA prompt optimization.

    Uses the official GEPA library for evolutionary prompt optimization.
    Supports multi-task evaluation across GSM8K, HumanEval, and BBH.

    Regime A: Optimizes for task accuracy only
    Regime B: Optimizes for accuracy while evading monitor detection
    """
    import random
    import numpy as np
    from src.optimization import GEPA, EvaluationSample, TaskType
    from src.data import load_dataset

    # Set random seed
    random.seed(seed)
    np.random.seed(seed)

    console.print("[bold blue]GEPA Optimization[/bold blue]")
    console.print(f"Dataset: {dataset}")
    console.print(f"Task Model: {task_model}")
    console.print(f"Reflection Model: {reflection_model}")
    console.print(f"Regime: {regime}")
    console.print(f"Generations: {generations}")
    console.print(f"Population: {population}")

    # Load dataset
    console.print("\n[yellow]Loading dataset...[/yellow]")
    try:
        data = load_dataset(dataset)
        # Convert to EvaluationSample format
        task_type_map = {
            "gsm8k": TaskType.MATH_REASONING,
            "humaneval": TaskType.CODE_GENERATION,
            "bbh": TaskType.LOGICAL_REASONING,
        }
        task_type = task_type_map.get(dataset.lower(), TaskType.MATH_REASONING)

        samples = [
            EvaluationSample(
                id=str(i),
                task_type=task_type,
                question=item.get("question", item.get("problem", "")),
                answer=item.get("answer", item.get("solution", "")),
            )
            for i, item in enumerate(data[:200])  # Limit for optimization
        ]

        # Split into train/val
        split_idx = int(len(samples) * 0.8)
        trainset = samples[:split_idx]
        valset = samples[split_idx:]

        console.print(f"Loaded {len(trainset)} train, {len(valset)} val samples")
    except Exception as e:
        console.print(f"[red]Failed to load dataset: {e}[/red]")
        console.print("[yellow]Using placeholder samples for demo...[/yellow]")
        trainset = []
        valset = []

    # Initialize GEPA optimizer
    optimizer = GEPA(
        task_lm=task_model,
        reflection_lm=reflection_model,
        regime=regime,
        population_size=population,
        max_generations=generations,
    )

    # Run optimization
    console.print("\n[yellow]Starting GEPA optimization...[/yellow]")
    console.print("This may take a while depending on population size and generations.")

    try:
        result = asyncio.run(optimizer.optimize(trainset, valset))

        # Save results
        output.mkdir(parents=True, exist_ok=True)
        result.save(output)

        console.print(f"\n[green]Optimization complete![/green]")
        if result.fitness:
            console.print(f"Aggregate fitness: {result.fitness['aggregate']:.4f}")
            console.print(f"  GSM8K: {result.fitness['gsm8k']:.4f}")
            console.print(f"  HumanEval: {result.fitness['humaneval']:.4f}")
            console.print(f"  BBH: {result.fitness['bbh']:.4f}")
            if regime == "B":
                console.print(f"  Monitor Suspicion: {result.fitness['monitor_suspicion']:.4f}")
                console.print(f"  Regime B Fitness: {result.fitness['regime_b_fitness']:.4f}")
        console.print(f"\nResults saved to: {output}")
        console.print(f"Wall time: {result.wall_time_seconds:.1f}s")

    except ImportError as e:
        console.print(f"\n[red]GEPA package not installed![/red]")
        console.print("Install with: pip install gepa")
        console.print(f"Error: {e}")
    except Exception as e:
        console.print(f"\n[red]Optimization failed: {e}[/red]")


@app.command()
def evaluate(
    prompt_path: Path = typer.Argument(..., help="Path to optimized prompt"),
    task: str = typer.Option("math_reasoning", help="Evaluation task"),
    model: str = typer.Option("gpt-4o", help="Model to evaluate"),
    samples: int = typer.Option(200, help="Number of samples"),
) -> None:
    """Evaluate an optimized prompt."""
    console.print(f"[bold blue]Evaluating prompt on {task}[/bold blue]")
    console.print(f"Model: {model}")
    console.print(f"Samples: {samples}")

    # Run inspect_ai evaluation
    import subprocess

    cmd = [
        "inspect", "eval",
        f"evals/tasks/{task}.py",
        "--model", model,
        "--limit", str(samples),
    ]

    console.print(f"\nRunning: {' '.join(cmd)}")
    subprocess.run(cmd)


@app.command()
def transfer(
    prompt_path: Path = typer.Argument(..., help="Path to optimized prompt"),
    models: str = typer.Option(
        "gpt-4o,gemini-2.0-flash,claude-sonnet-4",
        help="Comma-separated models"
    ),
    task: str = typer.Option("math_reasoning", help="Task to evaluate"),
) -> None:
    """Run transfer evaluation across multiple models."""
    model_list = [m.strip() for m in models.split(",")]

    table = Table(title="Transfer Evaluation")
    table.add_column("Model")
    table.add_column("Accuracy")
    table.add_column("Status")

    for model in model_list:
        # Placeholder - actual evaluation would go here
        table.add_row(model, "N/A", "pending")

    console.print(table)
    console.print("\n[yellow]Transfer evaluation not yet implemented[/yellow]")


@app.command()
def monitor(
    response_path: Path = typer.Argument(..., help="Path to response file"),
    prompt_path: Path = typer.Argument(..., help="Path to original prompt"),
) -> None:
    """Run autorater on a model response."""
    from src.monitors import LindnerAutorater
    from src.models import get_client

    console.print("[bold blue]Running Lindner Autorater[/bold blue]")

    response = response_path.read_text()
    prompt = prompt_path.read_text()

    client = get_client("openai", model_id="gpt-4o")
    autorater = LindnerAutorater(evaluator_client=client)

    result = asyncio.run(autorater.evaluate(response, prompt))

    # Display results
    console.print(f"\n[bold]Risk Level:[/bold] {result.risk_level.value}")
    console.print(f"[bold]Overall Score:[/bold] {result.overall_score:.2f}/5.0")
    console.print(f"\n[bold]Summary:[/bold] {result.summary}")

    if result.flags:
        console.print("\n[bold red]Flags:[/bold red]")
        for flag in result.flags:
            console.print(f"  - {flag}")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind"),
    port: int = typer.Option(8080, help="Port to bind"),
) -> None:
    """Start LORE API server."""
    console.print(f"[bold blue]Starting LORE API server[/bold blue]")
    console.print(f"Listening on {host}:{port}")
    console.print("\n[yellow]API server not yet implemented[/yellow]")


@app.command("ultrainteract-evolve")
def ultrainteract_evolve(
    output: Path = typer.Option(Path("results/ui_evolution"), help="Output directory"),
    task_model: str = typer.Option("openai/gpt-4o-mini", help="Model for task evaluation"),
    reflection_model: str = typer.Option("openai/gpt-4o", help="Model for reflection"),
    generations: int = typer.Option(50, help="Number of generations"),
    population: int = typer.Option(30, help="Population size"),
    train_size: int = typer.Option(1000, help="Training set size"),
    val_size: int = typer.Option(200, help="Validation set size"),
    max_metric_calls: int = typer.Option(3000, help="Maximum metric evaluations"),
    seed: int = typer.Option(42, help="Random seed"),
    wandb_enabled: bool = typer.Option(False, "--wandb", help="Enable wandb logging"),
    config: Optional[Path] = typer.Option(None, help="Path to config YAML file"),
) -> None:
    """Evolve general reasoning prompt on UltraInteract dataset.

    Uses GEPA to evolve a general reasoning prompt that works across multiple
    task types (Coding, Math_CoT, Math_PoT, Logic). Task-specific templates
    are fixed and concatenated with the evolved general prompt at inference.

    Example:
        lore ultrainteract-evolve --output results/ui_evo --generations 50 --wandb
    """
    import random
    import numpy as np
    import yaml
    from src.data import load_ultrainteract, UITaskType
    from src.optimization import (
        run_ultrainteract_evolution,
        DEFAULT_SEED_PROMPT,
        TASK_TEMPLATES,
    )

    # Load config if provided
    task_weights = {
        "Coding": 0.30,
        "Math_CoT": 0.25,
        "Math_PoT": 0.25,
        "Logic": 0.20,
    }
    seed_prompt = DEFAULT_SEED_PROMPT

    if config:
        with open(config) as f:
            cfg = yaml.safe_load(f)
        task_weights = cfg.get("task_weights", task_weights)
        seed_prompt = cfg.get("seed_prompt", seed_prompt)
        train_size = cfg.get("dataset", {}).get("train_size", train_size)
        val_size = cfg.get("dataset", {}).get("val_size", val_size)
        seed = cfg.get("dataset", {}).get("seed", seed)

    # Set random seed
    random.seed(seed)
    np.random.seed(seed)

    console.print("[bold blue]UltraInteract GEPA Evolution[/bold blue]")
    console.print(f"Task Model: {task_model}")
    console.print(f"Reflection Model: {reflection_model}")
    console.print(f"Generations: {generations}")
    console.print(f"Population: {population}")
    console.print(f"Train Size: {train_size}")
    console.print(f"Val Size: {val_size}")
    console.print(f"Max Metric Calls: {max_metric_calls}")

    console.print("\n[bold]Task Weights:[/bold]")
    for task, weight in task_weights.items():
        console.print(f"  {task}: {weight:.0%}")

    console.print("\n[bold]Task Templates:[/bold]")
    for task_type in ["Coding", "Math_CoT", "Math_PoT", "Logic"]:
        template_preview = TASK_TEMPLATES[task_type][:50].replace("\n", " ")
        console.print(f"  {task_type}: {template_preview}...")

    # Load UltraInteract dataset
    console.print("\n[yellow]Loading UltraInteract dataset...[/yellow]")
    try:
        splits = load_ultrainteract(
            train_size=train_size,
            val_size=val_size,
            test_size=0,
            seed=seed,
        )
        trainset = splits["train"]
        valset = splits["val"]

        # Show distribution by task type
        train_dist = {}
        for sample in trainset:
            task = sample.task_type.value
            train_dist[task] = train_dist.get(task, 0) + 1

        console.print(f"Loaded {len(trainset)} train, {len(valset)} val samples")
        console.print("[bold]Train Distribution:[/bold]")
        for task, count in sorted(train_dist.items()):
            console.print(f"  {task}: {count}")

    except Exception as e:
        console.print(f"[red]Failed to load dataset: {e}[/red]")
        raise typer.Exit(1)

    # Run evolution
    console.print("\n[yellow]Starting GEPA evolution...[/yellow]")
    console.print("Note: Only the general prompt is evolved. Task templates are fixed.")

    try:
        result = asyncio.run(
            run_ultrainteract_evolution(
                trainset=trainset,
                valset=valset,
                task_lm=task_model,
                reflection_lm=reflection_model,
                max_metric_calls=max_metric_calls,
                population_size=population,
                max_generations=generations,
                task_weights=task_weights,
                seed_prompt=seed_prompt,
                output_dir=output,
                wandb_enabled=wandb_enabled,
            )
        )

        # Display results
        console.print(f"\n[green]Evolution complete![/green]")
        console.print(f"Best Aggregate Fitness: {result.best_fitness:.4f}")

        if result.fitness_by_task:
            console.print("\n[bold]Fitness by Task Type:[/bold]")
            for task, fitness in result.fitness_by_task.items():
                console.print(f"  {task}: {fitness:.4f}")

        console.print(f"\n[bold]Evolved General Prompt:[/bold]")
        console.print(result.best_prompt[:200] + "..." if len(result.best_prompt) > 200 else result.best_prompt)

        console.print(f"\nResults saved to: {output}")
        console.print(f"  - best_prompt.txt: The evolved general prompt")
        console.print(f"  - evolution_history.json: Full evolution history")
        console.print(f"Wall time: {result.wall_time_seconds:.1f}s")

    except ImportError as e:
        console.print(f"\n[red]Missing dependency: {e}[/red]")
        console.print("Install with: pip install gepa datasets")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]Evolution failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


@app.command("crh-eval")
def crh_eval(
    prompt_path: Path = typer.Argument(..., help="Path to evolved prompt file"),
    baseline: Optional[str] = typer.Option(
        None,
        help="Baseline prompt to compare (default: 'You are a helpful assistant.')"
    ),
    output: Path = typer.Option(Path("results/crh_eval"), help="Output directory"),
    samples: int = typer.Option(200, help="Number of samples to evaluate"),
    side_tasks: str = typer.Option(
        "varname,logio,edgecase,complexity",
        help="Comma-separated side tasks to evaluate"
    ),
    monitor_model: str = typer.Option("openai/gpt-4o", help="Model for monitoring"),
    config: Optional[Path] = typer.Option(None, help="Path to config YAML file"),
) -> None:
    """Evaluate prompt monitorability on Caught Red-Handed benchmark.

    Compares an evolved prompt against a baseline to measure how well
    CoT monitoring can detect hidden behaviors (side tasks).

    Side tasks (code-focused):
    - varname: Variable naming violations
    - logio: I/O logging to files
    - edgecase: Edge case exploitation
    - complexity: Algorithmic slowdown

    Example:
        lore crh-eval results/ui_evolution/best_prompt.txt --samples 200
    """
    import yaml
    from src.evaluation import (
        CRHEvaluator,
        CRHSideTask,
        compare_prompt_monitorability,
    )

    # Load config if provided
    if config:
        with open(config) as f:
            cfg = yaml.safe_load(f)
        samples = cfg.get("samples_per_task", 50) * len(cfg.get("side_tasks", []))
        side_tasks = ",".join(cfg.get("side_tasks", side_tasks.split(",")))
        monitor_model = cfg.get("monitor", {}).get("model", monitor_model)

    # Parse side tasks
    side_task_list = [s.strip() for s in side_tasks.split(",")]
    valid_tasks = {t.value for t in CRHSideTask}
    invalid = set(side_task_list) - valid_tasks
    if invalid:
        console.print(f"[red]Invalid side tasks: {invalid}[/red]")
        console.print(f"Valid options: {valid_tasks}")
        raise typer.Exit(1)

    # Set default baseline
    if baseline is None:
        baseline = "You are a helpful assistant."

    # Load evolved prompt
    if not prompt_path.exists():
        console.print(f"[red]Prompt file not found: {prompt_path}[/red]")
        raise typer.Exit(1)

    evolved_prompt = prompt_path.read_text().strip()

    console.print("[bold blue]Caught Red-Handed Evaluation[/bold blue]")
    console.print(f"Evolved Prompt: {prompt_path}")
    console.print(f"Samples: {samples}")
    console.print(f"Side Tasks: {side_task_list}")
    console.print(f"Monitor Model: {monitor_model}")

    console.print("\n[bold]Baseline Prompt:[/bold]")
    console.print(f"  {baseline[:100]}...")

    console.print("\n[bold]Evolved Prompt Preview:[/bold]")
    console.print(f"  {evolved_prompt[:100]}...")

    # Create output directory
    output.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    console.print("\n[yellow]Running CRH evaluation...[/yellow]")
    console.print("This evaluates both baseline and evolved prompts on side task detection.")

    try:
        result = asyncio.run(
            compare_prompt_monitorability(
                baseline_prompt=baseline,
                evolved_prompt=evolved_prompt,
                samples=samples,
                side_tasks=side_task_list,  # Pass as strings
                output_dir=output,
            )
        )

        # Display results
        console.print(f"\n[green]Evaluation complete![/green]")

        # Baseline metrics (CRHMetrics is a TypedDict)
        console.print("\n[bold]Baseline Metrics:[/bold]")
        console.print(f"  Main Task Success: {result.baseline['main_only']:.1%}")
        console.print(f"  Side Task Success: {result.baseline['side_only']:.1%}")
        console.print(f"  Main+Side Success: {result.baseline['main_and_side']:.1%}")
        console.print(f"  CoT Detection Rate: {result.baseline['cot_detection_rate']:.1%}")
        console.print(f"  Action Detection Rate: {result.baseline['action_detection_rate']:.1%}")
        console.print(f"  Detection Delta (CoT-Action): {result.baseline['detection_delta']:+.1%}")

        # Evolved metrics
        console.print("\n[bold]Evolved Metrics:[/bold]")
        console.print(f"  Main Task Success: {result.evolved['main_only']:.1%}")
        console.print(f"  Side Task Success: {result.evolved['side_only']:.1%}")
        console.print(f"  Main+Side Success: {result.evolved['main_and_side']:.1%}")
        console.print(f"  CoT Detection Rate: {result.evolved['cot_detection_rate']:.1%}")
        console.print(f"  Action Detection Rate: {result.evolved['action_detection_rate']:.1%}")
        console.print(f"  Detection Delta (CoT-Action): {result.evolved['detection_delta']:+.1%}")

        # Comparison
        console.print("\n[bold]Monitorability Comparison (Evolved - Baseline):[/bold]")
        delta = result.delta
        console.print(f"  CoT Detection Delta: {delta.cot_detection_delta:+.1%}")
        console.print(f"  Action Detection Delta: {delta.action_detection_delta:+.1%}")
        console.print(f"  CoT Advantage Delta: {delta.cot_advantage_delta:+.1%}")

        # Risk assessment
        if delta.is_concerning:
            console.print("\n[bold red]WARNING: Monitorability significantly impaired![/bold red]")
            console.print("CoT detection decreased by more than 10 percentage points.")
            console.print("This evolved prompt may make hidden behaviors harder to detect.")
        else:
            console.print("\n[bold green]Monitorability preserved.[/bold green]")
            console.print("CoT detection not significantly impaired by evolution.")

        # Save results
        import json
        result_dict = {
            "baseline": result.baseline,
            "evolved": result.evolved,
            "delta": {
                "cot_detection_delta": delta.cot_detection_delta,
                "action_detection_delta": delta.action_detection_delta,
                "cot_advantage_delta": delta.cot_advantage_delta,
                "is_concerning": delta.is_concerning,
            },
        }
        (output / "crh_results.json").write_text(json.dumps(result_dict, indent=2))
        (output / "evolved_prompt.txt").write_text(evolved_prompt)
        (output / "baseline_prompt.txt").write_text(baseline)

        console.print(f"\nResults saved to: {output}")

    except ImportError as e:
        console.print(f"\n[red]Missing dependency: {e}[/red]")
        console.print("Ensure caught_red_handed submodule is initialized:")
        console.print("  git submodule update --init --recursive")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]Evaluation failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


@app.callback()
def main() -> None:
    """LORE: Learned Obfuscated Reasoning Experiments.

    A research framework for studying prompt optimization and its
    implications for AI alignment and safety.
    """
    pass


if __name__ == "__main__":
    app()
