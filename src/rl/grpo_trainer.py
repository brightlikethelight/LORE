"""GRPO Trainer for LORE.

This module provides a wrapper around TRL's GRPOTrainer for training
language models on the UltraInteract dataset with task-based rewards.

GRPO (Group Relative Policy Optimization) is a reinforcement learning
algorithm that uses group-based baselines instead of a learned value function,
making it more memory-efficient than PPO.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset

from src.rl.config import GRPOConfig, LoRAConfig, RLConfig
from src.rl.reward import TaskRewardFunction

logger = logging.getLogger(__name__)


from transformers import TrainerCallback


class AccuracyLoggingCallback(TrainerCallback):
    """Callback to log accuracy metrics during training."""

    def __init__(self, accuracy_tracker: "AccuracyTracker", baseline_accuracy: dict[str, float] | None = None):
        self.accuracy_tracker = accuracy_tracker
        self.baseline_accuracy = baseline_accuracy or {}

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log accuracy metrics alongside other metrics."""
        if logs is not None:
            # Add per-task accuracy to logs
            acc_metrics = self.accuracy_tracker.get_step_metrics(prefix="accuracy")
            for key, value in acc_metrics.items():
                logs[key] = value

            # Log cumulative accuracy
            cumulative = self.accuracy_tracker.get_metrics(prefix="cumulative_accuracy")
            for key, value in cumulative.items():
                logs[key] = value

            # Log improvement over baseline
            if self.baseline_accuracy:
                for task_type, baseline in self.baseline_accuracy.items():
                    current = self.accuracy_tracker.get_accuracy(task_type)
                    delta = current - baseline
                    logs[f"improvement/{task_type}"] = delta
                    logs[f"baseline/{task_type}"] = baseline

            # Reset step counters after logging
            self.accuracy_tracker.reset_step()


class AccuracyTracker:
    """Tracks accuracy metrics per task type during training."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all counters."""
        self._correct = defaultdict(int)
        self._total = defaultdict(int)
        self._step_correct = defaultdict(int)
        self._step_total = defaultdict(int)

    def update(self, task_type: str, is_correct: bool):
        """Update accuracy counts."""
        self._correct[task_type] += int(is_correct)
        self._total[task_type] += 1
        self._correct["overall"] += int(is_correct)
        self._total["overall"] += 1
        # Per-step tracking
        self._step_correct[task_type] += int(is_correct)
        self._step_total[task_type] += 1
        self._step_correct["overall"] += int(is_correct)
        self._step_total["overall"] += 1

    def get_accuracy(self, task_type: str = "overall") -> float:
        """Get cumulative accuracy for a task type."""
        if self._total[task_type] == 0:
            return 0.0
        return self._correct[task_type] / self._total[task_type]

    def get_step_accuracy(self, task_type: str = "overall") -> float:
        """Get accuracy for current step only."""
        if self._step_total[task_type] == 0:
            return 0.0
        return self._step_correct[task_type] / self._step_total[task_type]

    def reset_step(self):
        """Reset per-step counters (call after logging)."""
        self._step_correct.clear()
        self._step_total.clear()

    def get_metrics(self, prefix: str = "accuracy") -> dict[str, float]:
        """Get all accuracy metrics as a dict."""
        metrics = {}
        for task_type in self._total:
            if self._total[task_type] > 0:
                key = f"{prefix}/{task_type}"
                metrics[key] = self.get_accuracy(task_type)
        return metrics

    def get_step_metrics(self, prefix: str = "step_accuracy") -> dict[str, float]:
        """Get per-step accuracy metrics."""
        metrics = {}
        for task_type in self._step_total:
            if self._step_total[task_type] > 0:
                key = f"{prefix}/{task_type}"
                metrics[key] = self.get_step_accuracy(task_type)
        return metrics

    def get_counts(self) -> dict[str, dict[str, int]]:
        """Get raw counts for all task types."""
        return {
            task_type: {"correct": self._correct[task_type], "total": self._total[task_type]}
            for task_type in self._total
        }


@dataclass
class TrainingResult:
    """Result of GRPO training.

    Attributes:
        model_path: Path to saved model checkpoint
        final_loss: Final training loss
        training_steps: Total number of training steps
        wall_time_seconds: Total training time
        metrics: Training metrics history
        config: Configuration used for training
    """

    model_path: Path
    final_loss: float
    training_steps: int
    wall_time_seconds: float
    metrics: dict[str, Any]
    config: dict[str, Any]

    def save(self, path: Path) -> None:
        """Save training result to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        result_dict = {
            "model_path": str(self.model_path),
            "final_loss": self.final_loss,
            "training_steps": self.training_steps,
            "wall_time_seconds": self.wall_time_seconds,
            "metrics": self.metrics,
            "config": self.config,
            "timestamp": datetime.now().isoformat(),
        }

        with open(path / "training_result.json", "w") as f:
            json.dump(result_dict, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> TrainingResult:
        """Load training result from disk."""
        path = Path(path)
        with open(path / "training_result.json") as f:
            data = json.load(f)

        return cls(
            model_path=Path(data["model_path"]),
            final_loss=data["final_loss"],
            training_steps=data["training_steps"],
            wall_time_seconds=data["wall_time_seconds"],
            metrics=data["metrics"],
            config=data["config"],
        )


class LOREGRPOTrainer:
    """GRPO Trainer wrapper for LORE experiments.

    This class wraps TRL's GRPOTrainer to provide:
    - Task-based reward computation for UltraInteract
    - LoRA support for memory-efficient training
    - WandB integration for experiment tracking
    - Checkpoint management

    Example usage:
        config = RLConfig.from_yaml("configs/rl_config.yaml")
        trainer = LOREGRPOTrainer(config)

        # Load data
        from src.rl.data_pipeline import load_ultrainteract_for_grpo
        datasets = load_ultrainteract_for_grpo(
            train_size=5000,
            tokenizer=trainer.tokenizer,
        )

        # Train
        result = trainer.train(
            train_dataset=datasets["train"],
            eval_dataset=datasets["val"],
        )
    """

    def __init__(
        self,
        config: RLConfig | None = None,
        grpo_config: GRPOConfig | None = None,
        lora_config: LoRAConfig | None = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            config: Complete RL configuration
            grpo_config: GRPO-specific configuration (overrides config.grpo)
            lora_config: LoRA-specific configuration (overrides config.lora)
        """
        self.config = config or RLConfig()
        self.grpo_config = grpo_config or self.config.grpo
        self.lora_config = lora_config or self.config.lora

        # Lazy initialization
        self._model = None
        self._tokenizer = None
        self._trainer = None
        self._reward_fn = None

        # Accuracy tracking
        self.accuracy_tracker = AccuracyTracker()

    @property
    def model(self) -> Any:
        """Get the model, loading if necessary."""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def tokenizer(self) -> Any:
        """Get the tokenizer, loading if necessary."""
        if self._tokenizer is None:
            self._load_tokenizer()
        return self._tokenizer

    @property
    def reward_fn(self) -> TaskRewardFunction:
        """Get the reward function."""
        if self._reward_fn is None:
            self._reward_fn = TaskRewardFunction(
                correct_reward=self.config.reward.correct_reward,
                incorrect_reward=self.config.reward.incorrect_reward,
                math_tolerance=self.config.reward.math_tolerance,
                partial_credit=self.config.reward.partial_credit,
            )
        return self._reward_fn

    def _load_tokenizer(self) -> None:
        """Load the tokenizer."""
        from transformers import AutoTokenizer

        logger.info(f"Loading tokenizer: {self.grpo_config.model_name}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.grpo_config.model_name,
            trust_remote_code=True,
        )

        # Ensure pad token is set
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

    def _load_model(self) -> None:
        """Load the model with optional LoRA."""
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        logger.info(f"Loading model: {self.grpo_config.model_name}")

        # Configure quantization for memory efficiency
        bnb_config = None
        if torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        # Load base model
        self._model = AutoModelForCausalLM.from_pretrained(
            self.grpo_config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.grpo_config.bf16 else torch.float32,
        )

        # Apply LoRA if enabled
        if self.lora_config.enabled:
            from peft import get_peft_model, prepare_model_for_kbit_training

            logger.info("Applying LoRA adapter")

            # Prepare for training with quantization
            if bnb_config is not None:
                self._model = prepare_model_for_kbit_training(self._model)

            # Get PEFT config
            peft_config = self.lora_config.to_peft_config()

            # Apply LoRA
            self._model = get_peft_model(self._model, peft_config)
            self._model.print_trainable_parameters()

    def evaluate_baseline(
        self,
        eval_dataset: Dataset,
        max_samples: int = 200,
        max_new_tokens: int = 512,
    ) -> dict[str, float]:
        """Evaluate base model to get baseline accuracy before training.

        Args:
            eval_dataset: Dataset with 'prompt', 'ground_truth', 'task_type'
            max_samples: Maximum samples to evaluate
            max_new_tokens: Max tokens to generate

        Returns:
            Dict mapping task_type -> accuracy
        """
        logger.info("Evaluating baseline accuracy...")

        # Ensure model and tokenizer are loaded
        _ = self.model
        _ = self.tokenizer

        # Sample subset if needed
        if len(eval_dataset) > max_samples:
            eval_dataset = eval_dataset.shuffle(seed=42).select(range(max_samples))

        baseline_tracker = AccuracyTracker()

        # Generate and evaluate
        self._model.eval()
        with torch.no_grad():
            for i, sample in enumerate(eval_dataset):
                prompt = sample["prompt"]
                ground_truth = sample.get("ground_truth", "")
                task_type = sample.get("task_type", "Coding")

                # Tokenize
                inputs = self._tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                ).to(self._model.device)

                # Generate
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=self.grpo_config.temperature,
                    do_sample=True,
                    pad_token_id=self._tokenizer.pad_token_id,
                )

                # Decode
                completion = self._tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )

                # Evaluate
                result = self.reward_fn(completion, {"answer": ground_truth, "task_type": task_type})
                is_correct = result.reward > 0.5
                baseline_tracker.update(task_type, is_correct)

                if (i + 1) % 50 == 0:
                    logger.info(f"Baseline eval: {i + 1}/{len(eval_dataset)} samples")

        # Get baseline accuracies
        baseline_accuracy = {}
        for task_type in baseline_tracker._total:
            baseline_accuracy[task_type] = baseline_tracker.get_accuracy(task_type)

        logger.info("Baseline Accuracy by Task Type:")
        for task_type, acc in baseline_accuracy.items():
            counts = baseline_tracker.get_counts()[task_type]
            logger.info(f"  {task_type}: {acc:.2%} ({counts['correct']}/{counts['total']})")

        return baseline_accuracy

    def _create_reward_function(self) -> callable:
        """Create a reward function compatible with TRL's GRPOTrainer.

        Returns:
            Callable that takes (completions, prompts, **kwargs) and returns rewards
        """
        def reward_function(
            completions: list[str],
            prompts: list[str] | None = None,
            ground_truth: list[str] | None = None,
            task_type: list[str] | None = None,
            **kwargs: Any,
        ) -> list[float]:
            """Compute rewards for completions.

            Args:
                completions: List of model completions
                prompts: List of prompts (optional, not used)
                ground_truth: List of expected answers
                task_type: List of task types
                **kwargs: Additional arguments (ignored)

            Returns:
                List of reward values
            """
            rewards = []

            for i, completion in enumerate(completions):
                # Build sample dict for reward computation
                current_task_type = task_type[i] if task_type else "Coding"
                sample = {
                    "answer": ground_truth[i] if ground_truth else "",
                    "task_type": current_task_type,
                }

                # Compute reward
                result = self.reward_fn(completion, sample)
                rewards.append(result.reward)

                # Track accuracy per task type
                is_correct = result.reward > 0.5  # reward=1.0 for correct, 0.0 for incorrect
                self.accuracy_tracker.update(current_task_type, is_correct)

            return rewards

        return reward_function

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset | None = None,
        resume_from_checkpoint: str | Path | None = None,
        max_steps: int | None = None,
        evaluate_baseline: bool = True,
        baseline_samples: int = 100,
    ) -> TrainingResult:
        """Train the model using GRPO.

        Args:
            train_dataset: HuggingFace Dataset with 'prompt', 'ground_truth', 'task_type'
            eval_dataset: Optional evaluation dataset
            resume_from_checkpoint: Path to checkpoint to resume from
            max_steps: Maximum training steps (overrides config)
            evaluate_baseline: Whether to evaluate base model before training
            baseline_samples: Number of samples for baseline evaluation

        Returns:
            TrainingResult with model path and metrics
        """
        import time

        from trl import GRPOConfig as TRLGRPOConfig, GRPOTrainer

        start_time = time.time()

        # Ensure model and tokenizer are loaded
        _ = self.model
        _ = self.tokenizer

        # Evaluate baseline accuracy before training
        baseline_accuracy = {}
        if evaluate_baseline and eval_dataset is not None:
            baseline_accuracy = self.evaluate_baseline(
                eval_dataset,
                max_samples=baseline_samples,
            )

        # Create output directory
        output_dir = Path(self.grpo_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Configure TRL GRPO trainer
        # Note: GRPOConfig has specific parameters; use only supported ones
        training_args = TRLGRPOConfig(
            output_dir=str(output_dir),
            num_train_epochs=self.grpo_config.num_train_epochs,
            per_device_train_batch_size=self.grpo_config.per_device_train_batch_size,
            gradient_accumulation_steps=self.grpo_config.gradient_accumulation_steps,
            learning_rate=self.grpo_config.learning_rate,
            max_grad_norm=self.grpo_config.max_grad_norm,
            warmup_ratio=self.grpo_config.warmup_ratio,
            weight_decay=self.grpo_config.weight_decay,
            logging_steps=self.grpo_config.logging_steps,
            save_steps=self.grpo_config.save_steps,
            save_total_limit=self.grpo_config.save_total_limit,
            bf16=self.grpo_config.bf16,
            max_steps=max_steps if max_steps else -1,
            # GRPO-specific
            num_generations=self.grpo_config.num_generations,
            generation_batch_size=self.grpo_config.generation_batch_size,
            max_completion_length=self.grpo_config.max_new_tokens,
            temperature=self.grpo_config.temperature,
            beta=self.grpo_config.kl_coef,  # KL coefficient is called 'beta' in TRL
            # Reporting
            report_to=["wandb"] if self.config.wandb.enabled else ["none"],
        )

        # Initialize WandB if enabled
        if self.config.wandb.enabled:
            import wandb

            wandb.init(
                project=self.config.wandb.project,
                entity=self.config.wandb.entity,
                tags=self.config.wandb.tags,
                config=self.config.to_dict(),
            )

        # Create reward function
        reward_fn = self._create_reward_function()

        # Reset accuracy tracker for new training run
        self.accuracy_tracker.reset()

        # Create accuracy logging callback with baseline
        accuracy_callback = AccuracyLoggingCallback(
            self.accuracy_tracker,
            baseline_accuracy=baseline_accuracy,
        )

        # Create trainer
        logger.info("Initializing GRPOTrainer")
        trainer = GRPOTrainer(
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self._tokenizer,
            reward_funcs=reward_fn,
            callbacks=[accuracy_callback],
        )

        # Train
        logger.info("Starting GRPO training")
        train_result = trainer.train(
            resume_from_checkpoint=str(resume_from_checkpoint)
            if resume_from_checkpoint
            else None
        )

        # Save final model
        final_model_path = output_dir / "final"
        trainer.save_model(str(final_model_path))
        self._tokenizer.save_pretrained(str(final_model_path))
        logger.info(f"Saved final model to {final_model_path}")

        # Collect metrics
        wall_time = time.time() - start_time
        # train_result.metrics contains final values, not history lists
        metrics = {
            "train_loss": train_result.metrics.get("train_loss", 0.0),
            "train_runtime": train_result.metrics.get("train_runtime", 0.0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0.0),
        }

        # Add final accuracy metrics
        accuracy_metrics = self.accuracy_tracker.get_metrics(prefix="final_accuracy")
        metrics.update(accuracy_metrics)

        # Add per-task counts
        metrics["accuracy_counts"] = self.accuracy_tracker.get_counts()

        # Add baseline and improvement metrics
        if baseline_accuracy:
            metrics["baseline_accuracy"] = baseline_accuracy
            metrics["improvement"] = {}
            for task_type, baseline in baseline_accuracy.items():
                final = self.accuracy_tracker.get_accuracy(task_type)
                metrics["improvement"][task_type] = final - baseline

        # Log final accuracy summary with improvement
        logger.info("=" * 50)
        logger.info("TRAINING RESULTS - Accuracy by Task Type:")
        logger.info("=" * 50)
        for task_type, counts in self.accuracy_tracker.get_counts().items():
            final_acc = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
            baseline = baseline_accuracy.get(task_type, 0)
            delta = final_acc - baseline
            delta_str = f"+{delta:.1%}" if delta >= 0 else f"{delta:.1%}"
            if baseline > 0:
                logger.info(f"  {task_type:12}: {final_acc:.1%} (baseline: {baseline:.1%}, Δ: {delta_str})")
            else:
                logger.info(f"  {task_type:12}: {final_acc:.1%} ({counts['correct']}/{counts['total']})")
        logger.info("=" * 50)

        # Finish WandB
        if self.config.wandb.enabled:
            import wandb

            wandb.finish()

        result = TrainingResult(
            model_path=final_model_path,
            final_loss=train_result.training_loss,
            training_steps=train_result.global_step,
            wall_time_seconds=wall_time,
            metrics=metrics,
            config=self.config.to_dict(),
        )

        # Save result
        result.save(output_dir)

        logger.info(
            f"Training complete: {result.training_steps} steps, "
            f"loss={result.final_loss:.4f}, time={wall_time:.1f}s"
        )

        return result

    def save_model(self, path: Path | str) -> None:
        """Save the trained model.

        Args:
            path: Path to save the model
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self._model is not None:
            self._model.save_pretrained(path)
        if self._tokenizer is not None:
            self._tokenizer.save_pretrained(path)

        logger.info(f"Model saved to {path}")

    def load_model(self, path: Path | str) -> None:
        """Load a trained model from disk.

        Args:
            path: Path to load the model from
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = Path(path)

        logger.info(f"Loading model from {path}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=True,
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.grpo_config.bf16 else torch.float32,
        )


def train_grpo(
    config_path: Path | str | None = None,
    train_size: int = 5000,
    val_size: int = 500,
    max_steps: int | None = None,
    output_dir: Path | str | None = None,
    wandb_enabled: bool = False,
) -> TrainingResult:
    """Convenience function to run GRPO training.

    Args:
        config_path: Path to config YAML file
        train_size: Number of training samples
        val_size: Number of validation samples
        max_steps: Maximum training steps
        output_dir: Output directory (overrides config)
        wandb_enabled: Whether to enable WandB (overrides config)

    Returns:
        TrainingResult
    """
    from src.rl.config import load_config
    from src.rl.data_pipeline import load_ultrainteract_for_grpo

    # Load config
    config = load_config(config_path)

    # Override settings
    if output_dir:
        config.grpo.output_dir = str(output_dir)
    config.wandb.enabled = wandb_enabled
    config.dataset.train_size = train_size
    config.dataset.val_size = val_size

    # Create trainer
    trainer = LOREGRPOTrainer(config=config)

    # Load datasets
    logger.info("Loading UltraInteract dataset")
    datasets = load_ultrainteract_for_grpo(
        train_size=train_size,
        val_size=val_size,
        tokenizer=trainer.tokenizer,
        seed=config.dataset.seed,
        task_types=config.dataset.task_types,
    )

    # Train
    result = trainer.train(
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        max_steps=max_steps,
    )

    return result
