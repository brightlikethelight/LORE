#!/usr/bin/env python3
"""Training entry point for Verl GRPO on OpenR1-Math and SynLogic datasets.

This script provides a command-line interface for running GRPO training
using the Verl framework.

Usage:
    # Full training run
    python scripts/train_verl.py

    # Smoke test (100 samples, 10 steps)
    python scripts/train_verl.py --smoke-test

    # Custom configuration
    python scripts/train_verl.py --config configs/verl_grpo.yaml --output-dir results/my_run

    # With WandB logging
    python scripts/train_verl.py --wandb --wandb-project my-project

    # Math-only training
    python scripts/train_verl.py --math-samples 20000 --logic-samples 0 --code-samples 0
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rl.verl_trainer import VerlGRPOConfig, VerlGRPOTrainer, train_verl_grpo
from src.rl.verl_data_pipeline import load_verl_datasets
from src.rl.verl_rewards import create_verl_reward_function

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a model using Verl GRPO on OpenR1-Math and SynLogic datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Config and output
    parser.add_argument(
        "--config",
        type=str,
        default="configs/verl_grpo.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name or path (overrides config)",
    )

    # Dataset sizes
    parser.add_argument(
        "--math-samples",
        type=int,
        default=10000,
        help="Number of math training samples",
    )
    parser.add_argument(
        "--logic-samples",
        type=int,
        default=5000,
        help="Number of logic training samples",
    )
    parser.add_argument(
        "--code-samples",
        type=int,
        default=2000,
        help="Number of code training samples",
    )

    # Training parameters
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=16,
        help="GRPO group size (generations per prompt)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-6,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Prompt batch size",
    )

    # LoRA
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA (full fine-tuning)",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank",
    )

    # Logging
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable WandB logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="lore-verl-grpo",
        help="WandB project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="WandB entity (username or team)",
    )

    # Testing
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run smoke test (100 samples, 10 steps)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data and config but don't train",
    )

    # Code execution
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="Disable Docker for code execution (use in-process sandboxing)",
    )

    # Seed
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    # Checkpoint
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint path",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Handle smoke test mode
    if args.smoke_test:
        # Drastically reduced for serial generation fallback
        # 20 samples × 2 generations = 40 total generations
        # At ~40 sec/gen = ~30 min (fits in 3 hour limit with margin)
        logger.info("Running smoke test (20 samples, 5 steps)")
        args.math_samples = 10
        args.logic_samples = 6
        args.code_samples = 4
        args.max_steps = 5
        args.group_size = 2  # Minimal group size for GRPO
        args.output_dir = args.output_dir or "results/verl_smoke_test"

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        logger.info(f"Loading config from {config_path}")
        config = VerlGRPOConfig.from_yaml(config_path)
    else:
        logger.info("Using default config")
        config = VerlGRPOConfig()

    # Override config with command line arguments
    if args.model:
        config.model_name = args.model
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.group_size:
        config.group_size = args.group_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.batch_size:
        config.prompt_batch_size = args.batch_size
    if args.no_lora:
        config.use_lora = False
    if args.lora_r:
        config.lora_r = args.lora_r

    # Reduce max_response_length for smoke test to avoid OOM
    if args.smoke_test:
        config.max_response_length = 2048  # Shorter responses for smoke test

    # Log configuration
    logger.info("=" * 60)
    logger.info("Verl GRPO Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Math samples: {args.math_samples}")
    logger.info(f"Logic samples: {args.logic_samples}")
    logger.info(f"Code samples: {args.code_samples}")
    logger.info(f"Group size: {config.group_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Max steps: {args.max_steps or 'auto'}")
    logger.info(f"LoRA enabled: {config.use_lora}")
    logger.info(f"WandB enabled: {args.wandb}")
    logger.info("=" * 60)

    # Initialize WandB if enabled
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                config=config.to_dict(),
                tags=["grpo", "verl", "openr1", "synlogic"],
            )
            logger.info(f"WandB initialized: {wandb.run.url}")
        except ImportError:
            logger.warning("WandB not installed, disabling logging")

    # Create reward function
    logger.info("Creating reward function...")
    reward_fn = create_verl_reward_function(
        use_docker=not args.no_docker,
        partial_credit=False,
    )

    # Create trainer
    logger.info("Initializing trainer...")
    trainer = VerlGRPOTrainer(config=config, reward_fn=reward_fn)

    # Load datasets
    logger.info("Loading datasets...")
    datasets = load_verl_datasets(
        train_math=args.math_samples,
        train_logic=args.logic_samples,
        train_code=args.code_samples,
        val_math=max(args.math_samples // 20, 50),
        val_logic=max(args.logic_samples // 20, 25),
        val_code=max(args.code_samples // 20, 10),
        seed=args.seed,
        tokenizer=trainer.tokenizer,
    )

    logger.info(f"Train dataset size: {len(datasets['train'])}")
    logger.info(f"Validation dataset size: {len(datasets['val'])}")

    # Show sample distribution
    train_data = datasets["train"]
    task_counts = {}
    for item in train_data:
        task_type = item.get("task_type", "unknown")
        task_counts[task_type] = task_counts.get(task_type, 0) + 1
    logger.info(f"Task distribution: {task_counts}")

    # Dry run - just verify loading works
    if args.dry_run:
        logger.info("Dry run complete - data loaded successfully")
        logger.info(f"Sample prompt: {train_data[0]['prompt'][:200]}...")
        logger.info(f"Sample answer: {train_data[0]['answer'][:100]}...")
        return 0

    # Train
    logger.info("Starting training...")
    result = trainer.train(
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        max_steps=args.max_steps,
        resume_from_checkpoint=args.resume,
    )

    # Log results
    logger.info("=" * 60)
    logger.info("Training Complete")
    logger.info("=" * 60)
    logger.info(f"Model saved to: {result.model_path}")
    logger.info(f"Final loss: {result.final_loss:.4f}")
    logger.info(f"Training steps: {result.training_steps}")
    logger.info(f"Wall time: {result.wall_time_seconds:.1f}s")
    logger.info("=" * 60)

    # Finish WandB
    if args.wandb:
        try:
            import wandb
            wandb.log({
                "final_loss": result.final_loss,
                "training_steps": result.training_steps,
                "wall_time_seconds": result.wall_time_seconds,
            })
            wandb.finish()
        except:
            pass

    return 0


def test_data_loading():
    """Quick test of data loading functionality."""
    logger.info("Testing data loading...")

    # Test OpenR1-Math loading
    logger.info("Testing OpenR1-Math-220k loader...")
    try:
        from src.rl.verl_data_pipeline import load_openr1_math
        math_samples = load_openr1_math(sample_size=10)
        logger.info(f"Loaded {len(math_samples)} math samples")
        if math_samples:
            logger.info(f"Sample: {math_samples[0].prompt[:100]}...")
            logger.info(f"Answer: {math_samples[0].answer}")
    except Exception as e:
        logger.warning(f"OpenR1-Math loading failed: {e}")

    # Test SynLogic loading
    logger.info("Testing SynLogic loader...")
    try:
        from src.rl.verl_data_pipeline import load_synlogic
        logic_samples = load_synlogic(difficulty="easy", sample_size=10)
        logger.info(f"Loaded {len(logic_samples)} logic samples")
        if logic_samples:
            logger.info(f"Task: {logic_samples[0].extra_info.get('task_name', 'unknown')}")
    except Exception as e:
        logger.warning(f"SynLogic loading failed: {e}")

    # Test reward functions
    logger.info("Testing reward functions...")
    try:
        from src.rl.verl_rewards import MathVerifyReward, SynLogicReward

        math_reward = MathVerifyReward()
        result = math_reward.compute(
            "The answer is \\boxed{42}",
            {"answer": "42"}
        )
        logger.info(f"Math reward test: {result}")

        logic_reward = SynLogicReward()
        result = logic_reward.compute(
            "<think>Let me think...</think>\n<answer>yes</answer>",
            {"answer": "yes"}
        )
        logger.info(f"Logic reward test: {result}")
    except Exception as e:
        logger.warning(f"Reward function test failed: {e}")

    logger.info("Data loading tests complete")


if __name__ == "__main__":
    # Add test subcommand
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_data_loading()
        sys.exit(0)

    sys.exit(main())
