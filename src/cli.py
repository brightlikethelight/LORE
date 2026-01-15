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


@app.callback()
def main() -> None:
    """LORE: Learned Obfuscated Reasoning Experiments.

    A research framework for studying prompt optimization and its
    implications for AI alignment and safety.
    """
    pass


if __name__ == "__main__":
    app()
