#!/usr/bin/env python3
"""Training entry point for SkyRL GRPO.

This script provides a command-line interface for running SkyRL-based
GRPO training on multi-task reasoning datasets.

Usage:
    # Smoke test (quick validation)
    python scripts/train_skyrl.py --smoke-test

    # Full training
    python scripts/train_skyrl.py --model Qwen/Qwen2.5-7B-Instruct --math-samples 10000

    # With config file
    python scripts/train_skyrl.py --config configs/skyrl_grpo.yaml

    # With WandB logging
    python scripts/train_skyrl.py --wandb --smoke-test
"""

import logging
import random
import sys
from pathlib import Path

import click
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--model",
    default="Qwen/Qwen2.5-7B-Instruct",
    help="HuggingFace model name or path",
)
@click.option(
    "--output",
    default="results/skyrl_grpo",
    type=click.Path(),
    help="Output directory for checkpoints",
)
@click.option(
    "--math-samples",
    default=10000,
    type=int,
    help="Number of math training samples",
)
@click.option(
    "--logic-samples",
    default=5000,
    type=int,
    help="Number of logic training samples",
)
@click.option(
    "--code-samples",
    default=2000,
    type=int,
    help="Number of code training samples",
)
@click.option(
    "--group-size",
    default=4,
    type=int,
    help="GRPO group size (generations per prompt)",
)
@click.option(
    "--batch-size",
    default=8,
    type=int,
    help="Training batch size",
)
@click.option(
    "--max-steps",
    default=None,
    type=int,
    help="Maximum training steps (None for full epochs)",
)
@click.option(
    "--lr",
    default=1e-6,
    type=float,
    help="Learning rate",
)
@click.option(
    "--seed",
    default=42,
    type=int,
    help="Random seed",
)
@click.option(
    "--config",
    default=None,
    type=click.Path(exists=True),
    help="Path to YAML config file",
)
@click.option(
    "--wandb",
    is_flag=True,
    help="Enable Weights & Biases logging",
)
@click.option(
    "--smoke-test",
    is_flag=True,
    help="Run a quick smoke test with minimal samples",
)
@click.option(
    "--use-vllm/--no-vllm",
    default=True,
    help="Use vLLM for faster generation",
)
@click.option(
    "--use-docker/--no-docker",
    default=False,
    help="Use Docker for code execution (requires Docker)",
)
@click.option(
    "--load-in-8bit/--no-8bit",
    default=True,
    help="Use 8-bit quantization (disable for smaller models or better stability)",
)
@click.option(
    "--gradient-checkpointing/--no-gradient-checkpointing",
    default=True,
    help="Use gradient checkpointing (disable for debugging)",
)
@click.option(
    "--max-response-length",
    default=4096,
    type=int,
    help="Maximum response length for generation",
)
def main(
    model: str,
    output: str,
    math_samples: int,
    logic_samples: int,
    code_samples: int,
    group_size: int,
    batch_size: int,
    max_steps: int | None,
    lr: float,
    seed: int,
    config: str | None,
    wandb: bool,
    smoke_test: bool,
    use_vllm: bool,
    use_docker: bool,
    load_in_8bit: bool,
    gradient_checkpointing: bool,
    max_response_length: int,
) -> None:
    """Train a model on multi-task reasoning using SkyRL GRPO.

    SkyRL provides faster single-GPU training compared to Verl through
    better async batched generation. This is especially beneficial
    when running without a Ray cluster.
    """
    from src.rl.skyrl_trainer import SkyRLConfig, LORESkyRLTrainer
    from src.rl.skyrl_data_pipeline import load_skyrl_datasets, load_smoke_test_datasets
    from src.rl.skyrl_rewards import create_skyrl_reward_fn

    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)

    # Override settings for smoke test
    if smoke_test:
        logger.info("Running smoke test with minimal configuration")
        math_samples = 50
        logic_samples = 30
        code_samples = 20
        max_steps = 10
        batch_size = 2
        group_size = 2
        output_path = Path(output)
        output = str(output_path.parent / f"{output_path.name}_smoke")

    logger.info("=" * 60)
    logger.info("SkyRL GRPO Training")
    logger.info("=" * 60)
    logger.info(f"Model: {model}")
    logger.info(f"Output: {output}")
    logger.info(f"Math samples: {math_samples}")
    logger.info(f"Logic samples: {logic_samples}")
    logger.info(f"Code samples: {code_samples}")
    logger.info(f"Group size: {group_size}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Max steps: {max_steps or 'full epochs'}")
    logger.info(f"Learning rate: {lr}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Use vLLM: {use_vllm}")
    logger.info(f"Use Docker: {use_docker}")
    logger.info(f"WandB: {'enabled' if wandb else 'disabled'}")
    logger.info("=" * 60)

    # Load or create config
    if config:
        skyrl_config = SkyRLConfig.from_yaml(config)
        logger.info(f"Loaded config from {config}")
    else:
        skyrl_config = SkyRLConfig()

    # Override with CLI arguments
    skyrl_config.model_name = model
    skyrl_config.output_dir = output
    skyrl_config.group_size = group_size
    skyrl_config.batch_size = batch_size
    skyrl_config.learning_rate = lr
    skyrl_config.use_vllm = use_vllm
    skyrl_config.load_in_8bit = load_in_8bit
    skyrl_config.use_gradient_checkpointing = gradient_checkpointing
    skyrl_config.max_response_length = max_response_length

    # Create trainer
    logger.info("Initializing SkyRL trainer...")
    reward_fn = create_skyrl_reward_fn(use_docker=use_docker)
    trainer = LORESkyRLTrainer(config=skyrl_config, reward_fn=reward_fn)

    # Load datasets
    logger.info("Loading datasets...")
    if smoke_test:
        datasets = load_smoke_test_datasets(
            math_samples=math_samples,
            logic_samples=logic_samples,
            code_samples=code_samples,
            seed=seed,
            tokenizer=trainer.tokenizer,
        )
    else:
        datasets = load_skyrl_datasets(
            math_samples=math_samples,
            logic_samples=logic_samples,
            code_samples=code_samples,
            seed=seed,
            tokenizer=trainer.tokenizer,
        )

    logger.info(f"Train dataset: {len(datasets['train'])} samples")
    logger.info(f"Val dataset: {len(datasets['val'])} samples")

    # Show task distribution
    train_data = list(datasets["train"])
    task_counts: dict[str, int] = {}
    for sample in train_data:
        task_type = sample.get("task_type", "unknown")
        task_counts[task_type] = task_counts.get(task_type, 0) + 1

    logger.info("Task distribution:")
    for task, count in sorted(task_counts.items()):
        logger.info(f"  {task}: {count} ({100*count/len(train_data):.1f}%)")

    # Initialize WandB
    if wandb:
        import wandb as wandb_lib
        wandb_lib.init(
            project="lore-skyrl-grpo",
            config={
                **skyrl_config.to_dict(),
                "math_samples": math_samples,
                "logic_samples": logic_samples,
                "code_samples": code_samples,
                "smoke_test": smoke_test,
            },
            tags=["skyrl", "grpo"] + (["smoke-test"] if smoke_test else []),
        )

    # Train
    logger.info("Starting SkyRL GRPO training...")
    try:
        result = trainer.train(
            train_dataset=datasets["train"],
            eval_dataset=datasets["val"],
            max_steps=max_steps,
        )

        # Log results
        logger.info("=" * 60)
        logger.info("Training Complete!")
        logger.info("=" * 60)
        logger.info(f"Model saved to: {result.model_path}")
        logger.info(f"Training steps: {result.training_steps}")
        logger.info(f"Final loss: {result.final_loss:.4f}")
        logger.info(f"Wall time: {result.wall_time_seconds:.1f}s")

        if result.metrics:
            if "reward_mean" in result.metrics and result.metrics["reward_mean"]:
                final_reward = result.metrics["reward_mean"][-1]
                logger.info(f"Final mean reward: {final_reward:.4f}")

        # Log to WandB
        if wandb:
            wandb_lib.log({
                "final_loss": result.final_loss,
                "training_steps": result.training_steps,
                "wall_time_seconds": result.wall_time_seconds,
            })
            wandb_lib.finish()

        logger.info(f"Results saved to: {output}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        if wandb:
            import wandb as wandb_lib
            wandb_lib.finish(exit_code=1)
        sys.exit(1)

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        if wandb:
            import wandb as wandb_lib
            wandb_lib.finish(exit_code=1)
        sys.exit(1)


if __name__ == "__main__":
    main()
