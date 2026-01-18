#!/usr/bin/env python3
"""Run GEPA evolution with unlimited metric calls budget."""

import os
os.environ['HF_HOME'] = '/home/mila/t/thomas.jiralerspong/.cache/huggingface'

import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Setup logging
log_file = Path("/home/mila/t/thomas.jiralerspong/LORE/logs/evolution_v5_unlimited.log")
log_file.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def evaluate_prompt_on_samples(adapter, prompt, samples, name=""):
    """Evaluate a prompt on samples and return per-sample results."""
    candidate = {"general_prompt": prompt}
    eval_result = adapter.evaluate(samples, candidate, capture_traces=True)

    # Collect per-sample scores
    scores = eval_result.scores

    # Group by task type
    by_task = defaultdict(list)
    for i, sample in enumerate(samples):
        task_type = sample['additional_context']['task_type']
        by_task[task_type].append(scores[i])

    # Compute stats
    total_correct = sum(scores)
    total = len(scores)
    accuracy = total_correct / total if total > 0 else 0

    task_accuracies = {}
    for task_type, task_scores in by_task.items():
        task_accuracies[task_type] = sum(task_scores) / len(task_scores) if task_scores else 0

    logger.info(f"{name} Results: {total_correct}/{total} = {accuracy:.1%}")
    for task_type, acc in sorted(task_accuracies.items()):
        logger.info(f"  {task_type}: {acc:.1%} ({sum(by_task[task_type])}/{len(by_task[task_type])})")

    return {
        "scores": scores,
        "accuracy": accuracy,
        "task_accuracies": task_accuracies,
        "total_correct": total_correct,
        "total": total,
    }

async def main():
    logger.info(f"Starting GEPA Evolution v5 (unlimited budget) at {datetime.now()}")

    from src.data.ultrainteract import UltraInteractLoader, create_gepa_dataset
    from src.optimization.ultrainteract_gepa import (
        run_ultrainteract_evolution,
        UltraInteractGEPAAdapter,
        DEFAULT_SEED_PROMPT,
    )

    # Config - UNLIMITED budget to allow full evolution
    task_lm = "anthropic/claude-opus-4-5-20251101"
    train_size = 200      # Training samples
    val_size = 150        # Validation set for statistical power
    test_size = 100       # Held-out test set

    # With 150 val samples, each generation costs ~150 metric calls
    # Setting to 50,000 allows ~300+ generations (effectively unlimited)
    max_metric_calls = 50000

    logger.info(f"Config: {task_lm}")
    logger.info(f"Sizes: train={train_size}, val={val_size}, test={test_size}")
    logger.info(f"Budget: {max_metric_calls} metric calls (effectively unlimited)")

    # Load data with held-out test set
    logger.info("Loading UltraInteract dataset...")
    loader = UltraInteractLoader(seed=42)
    splits = loader.load_stratified(
        train_size=train_size,
        val_size=val_size,
        test_size=test_size
    )

    train_dist = loader.get_task_distribution(splits["train"])
    val_dist = loader.get_task_distribution(splits["val"])
    test_dist = loader.get_task_distribution(splits["test"])

    logger.info(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    logger.info(f"Train dist: {train_dist}")
    logger.info(f"Val dist: {val_dist}")
    logger.info(f"Test dist: {test_dist}")

    # Convert to GEPA format
    train_data = create_gepa_dataset(splits['train'])
    val_data = create_gepa_dataset(splits['val'])
    test_data = create_gepa_dataset(splits['test'])

    # Create adapter
    adapter = UltraInteractGEPAAdapter(task_lm=task_lm)

    # =========================================================================
    # BASELINE EVALUATION (before evolution)
    # =========================================================================
    logger.info("="*60)
    logger.info("BASELINE EVALUATION (Seed Prompt)")
    logger.info("="*60)

    baseline_val_results = await evaluate_prompt_on_samples(
        adapter, DEFAULT_SEED_PROMPT, val_data, "Baseline (Val)"
    )

    baseline_test_results = await evaluate_prompt_on_samples(
        adapter, DEFAULT_SEED_PROMPT, test_data, "Baseline (Test)"
    )

    # =========================================================================
    # RUN EVOLUTION
    # =========================================================================
    logger.info("="*60)
    logger.info("STARTING GEPA EVOLUTION (UNLIMITED BUDGET)")
    logger.info("="*60)

    output_dir = Path("/home/mila/t/thomas.jiralerspong/LORE/results/ui_evolution_v5")

    result = await run_ultrainteract_evolution(
        trainset=splits["train"],
        valset=splits["val"],
        task_lm=task_lm,
        reflection_lm=task_lm,
        max_metric_calls=max_metric_calls,
        seed_prompt=DEFAULT_SEED_PROMPT,
        output_dir=output_dir,
        use_wandb=False,
        seed=42,
    )

    evolved_prompt = result.best_general_prompt

    logger.info("="*60)
    logger.info(f"EVOLUTION COMPLETE: {result.generations_completed} generations, {result.total_evaluations} evaluations")
    logger.info("="*60)
    logger.info(f"Evolved prompt:\n{evolved_prompt}")

    # =========================================================================
    # EVOLVED PROMPT EVALUATION
    # =========================================================================
    logger.info("="*60)
    logger.info("EVOLVED PROMPT EVALUATION")
    logger.info("="*60)

    evolved_val_results = await evaluate_prompt_on_samples(
        adapter, evolved_prompt, val_data, "Evolved (Val)"
    )

    evolved_test_results = await evaluate_prompt_on_samples(
        adapter, evolved_prompt, test_data, "Evolved (Test)"
    )

    # =========================================================================
    # STATISTICAL ANALYSIS
    # =========================================================================
    logger.info("="*60)
    logger.info("STATISTICAL ANALYSIS")
    logger.info("="*60)

    # McNemar's test on validation set
    baseline_scores = baseline_val_results["scores"]
    evolved_scores = evolved_val_results["scores"]

    # Count discordant pairs
    b = sum(1 for i in range(len(baseline_scores))
            if baseline_scores[i] == 0 and evolved_scores[i] == 1)  # wrong->correct
    c = sum(1 for i in range(len(baseline_scores))
            if baseline_scores[i] == 1 and evolved_scores[i] == 0)  # correct->wrong

    logger.info(f"Validation Set (n={len(baseline_scores)}):")
    logger.info(f"  Baseline: {baseline_val_results['accuracy']:.1%}")
    logger.info(f"  Evolved:  {evolved_val_results['accuracy']:.1%}")
    logger.info(f"  Improvement: {evolved_val_results['accuracy'] - baseline_val_results['accuracy']:.1%}")
    logger.info(f"  Discordant pairs: b={b} (wrong->correct), c={c} (correct->wrong)")

    # McNemar's test
    if b + c > 0:
        if b + c < 25:
            # Use exact binomial test for small samples
            from math import comb
            # P(X >= b) under H0: X ~ Binomial(b+c, 0.5)
            p_value = sum(comb(b+c, k) * 0.5**(b+c) for k in range(b, b+c+1))
            logger.info(f"  McNemar's exact test p-value: {p_value:.4f}")
        else:
            # Use chi-squared approximation
            import math
            chi2 = (abs(b - c) - 1)**2 / (b + c)
            # Approximate p-value (one-tailed)
            p_value = 0.5 * math.erfc(math.sqrt(chi2/2))
            logger.info(f"  McNemar's chi-squared: {chi2:.2f}, p-value: {p_value:.4f}")
    else:
        logger.info("  No discordant pairs - cannot compute McNemar's test")
        p_value = 1.0

    # Test set results (held-out)
    logger.info(f"\nHeld-out Test Set (n={len(test_data)}):")
    logger.info(f"  Baseline: {baseline_test_results['accuracy']:.1%}")
    logger.info(f"  Evolved:  {evolved_test_results['accuracy']:.1%}")
    logger.info(f"  Improvement: {evolved_test_results['accuracy'] - baseline_test_results['accuracy']:.1%}")

    # Statistical significance
    is_significant = p_value < 0.05
    logger.info(f"\nStatistically significant (p < 0.05): {is_significant}")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    full_results = {
        "config": {
            "train_size": len(train_data),
            "val_size": len(val_data),
            "test_size": len(test_data),
            "max_metric_calls": max_metric_calls,
            "task_lm": task_lm,
        },
        "baseline": {
            "val_accuracy": baseline_val_results["accuracy"],
            "test_accuracy": baseline_test_results["accuracy"],
            "val_task_accuracies": baseline_val_results["task_accuracies"],
            "test_task_accuracies": baseline_test_results["task_accuracies"],
        },
        "evolved": {
            "val_accuracy": evolved_val_results["accuracy"],
            "test_accuracy": evolved_test_results["accuracy"],
            "val_task_accuracies": evolved_val_results["task_accuracies"],
            "test_task_accuracies": evolved_test_results["task_accuracies"],
            "prompt": evolved_prompt,
        },
        "statistical_analysis": {
            "val_improvement": evolved_val_results["accuracy"] - baseline_val_results["accuracy"],
            "test_improvement": evolved_test_results["accuracy"] - baseline_test_results["accuracy"],
            "discordant_b": b,
            "discordant_c": c,
            "p_value": p_value,
            "is_significant": is_significant,
        },
        "evolution_stats": {
            "generations": result.generations_completed,
            "total_evaluations": result.total_evaluations,
            "wall_time_seconds": result.wall_time_seconds,
        },
        "timestamp": datetime.now().isoformat(),
    }

    results_path = output_dir / "full_statistical_results.json"
    with open(results_path, "w") as f:
        json.dump(full_results, f, indent=2)

    logger.info(f"\nFull results saved to: {results_path}")

    # Final summary
    logger.info("="*60)
    logger.info("FINAL SUMMARY")
    logger.info("="*60)
    logger.info(f"Generations completed: {result.generations_completed}")
    logger.info(f"Total evaluations: {result.total_evaluations}")
    logger.info(f"Wall time: {result.wall_time_seconds/3600:.1f} hours")
    logger.info(f"Validation (n={len(val_data)}): {baseline_val_results['accuracy']:.1%} -> {evolved_val_results['accuracy']:.1%} (p={p_value:.4f})")
    logger.info(f"Test (n={len(test_data)}): {baseline_test_results['accuracy']:.1%} -> {evolved_test_results['accuracy']:.1%}")
    logger.info(f"Significant: {is_significant}")

    return full_results

if __name__ == "__main__":
    asyncio.run(main())
