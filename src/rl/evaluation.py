"""Evaluation module for RL-trained models.

This module provides evaluation pipelines for comparing RL-trained models
against base models on the UltraInteract dataset.

Key metrics:
- Overall accuracy
- Per-task accuracy (Coding, Math_CoT, Math_PoT, Logic)
- Response quality metrics
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
from tqdm import tqdm

from src.data.ultrainteract import (
    UITaskType,
    UltraInteractLoader,
    UltraInteractSample,
)
from src.rl.data_pipeline import format_chat_prompt
from src.rl.reward import TaskRewardFunction, RewardResult

logger = logging.getLogger(__name__)


@dataclass
class EvaluationSample:
    """A single evaluation sample with prediction and result.

    Attributes:
        sample_id: Original sample ID
        task_type: Task type
        prompt: Input prompt
        response: Model's response
        expected: Expected answer
        is_correct: Whether the response is correct
        reward: Reward value
        feedback: Grading feedback
    """

    sample_id: str
    task_type: str
    prompt: str
    response: str
    expected: str
    is_correct: bool
    reward: float
    feedback: str


@dataclass
class RLEvaluationResult:
    """Complete evaluation result.

    Attributes:
        model_path: Path to evaluated model
        overall_accuracy: Overall accuracy across all tasks
        per_task_accuracy: Accuracy per task type
        total_samples: Total number of samples
        samples_by_task: Number of samples per task type
        evaluation_samples: List of individual evaluation samples
        wall_time_seconds: Total evaluation time
        config: Evaluation configuration
    """

    model_path: str
    overall_accuracy: float
    per_task_accuracy: dict[str, float]
    total_samples: int
    samples_by_task: dict[str, int]
    evaluation_samples: list[EvaluationSample] = field(default_factory=list)
    wall_time_seconds: float = 0.0
    config: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path | str) -> None:
        """Save evaluation result to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save summary
        summary = {
            "model_path": self.model_path,
            "overall_accuracy": self.overall_accuracy,
            "per_task_accuracy": self.per_task_accuracy,
            "total_samples": self.total_samples,
            "samples_by_task": self.samples_by_task,
            "wall_time_seconds": self.wall_time_seconds,
            "config": self.config,
            "timestamp": datetime.now().isoformat(),
        }

        with open(path / "evaluation_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Save detailed samples
        samples_data = [
            {
                "sample_id": s.sample_id,
                "task_type": s.task_type,
                "response": s.response[:500],  # Truncate for storage
                "expected": s.expected[:200],
                "is_correct": s.is_correct,
                "reward": s.reward,
                "feedback": s.feedback,
            }
            for s in self.evaluation_samples
        ]

        with open(path / "evaluation_samples.json", "w") as f:
            json.dump(samples_data, f, indent=2)

        logger.info(f"Saved evaluation results to {path}")

    @classmethod
    def load(cls, path: Path | str) -> RLEvaluationResult:
        """Load evaluation result from disk."""
        path = Path(path)

        with open(path / "evaluation_summary.json") as f:
            summary = json.load(f)

        samples = []
        samples_path = path / "evaluation_samples.json"
        if samples_path.exists():
            with open(samples_path) as f:
                samples_data = json.load(f)
            samples = [
                EvaluationSample(
                    sample_id=s["sample_id"],
                    task_type=s["task_type"],
                    prompt="",  # Not stored
                    response=s["response"],
                    expected=s["expected"],
                    is_correct=s["is_correct"],
                    reward=s["reward"],
                    feedback=s["feedback"],
                )
                for s in samples_data
            ]

        return cls(
            model_path=summary["model_path"],
            overall_accuracy=summary["overall_accuracy"],
            per_task_accuracy=summary["per_task_accuracy"],
            total_samples=summary["total_samples"],
            samples_by_task=summary["samples_by_task"],
            evaluation_samples=samples,
            wall_time_seconds=summary.get("wall_time_seconds", 0.0),
            config=summary.get("config", {}),
        )


@dataclass
class ModelComparisonResult:
    """Result of comparing two models.

    Attributes:
        base_result: Evaluation result for base model
        trained_result: Evaluation result for trained model
        accuracy_delta: Overall accuracy improvement
        per_task_delta: Per-task accuracy improvement
    """

    base_result: RLEvaluationResult
    trained_result: RLEvaluationResult
    accuracy_delta: float
    per_task_delta: dict[str, float]

    def save(self, path: Path | str) -> None:
        """Save comparison result."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save base and trained results
        self.base_result.save(path / "base")
        self.trained_result.save(path / "trained")

        # Save comparison summary
        comparison = {
            "accuracy_delta": self.accuracy_delta,
            "per_task_delta": self.per_task_delta,
            "base_accuracy": self.base_result.overall_accuracy,
            "trained_accuracy": self.trained_result.overall_accuracy,
            "base_per_task": self.base_result.per_task_accuracy,
            "trained_per_task": self.trained_result.per_task_accuracy,
            "timestamp": datetime.now().isoformat(),
        }

        with open(path / "comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)


class RLModelEvaluator:
    """Evaluator for RL-trained models.

    This class provides evaluation functionality for comparing model performance
    on the UltraInteract dataset.
    """

    def __init__(
        self,
        reward_config: dict[str, Any] | None = None,
        batch_size: int = 8,
        max_new_tokens: int = 512,
        temperature: float = 0.0,  # Greedy for evaluation
    ) -> None:
        """Initialize the evaluator.

        Args:
            reward_config: Configuration for reward function
            batch_size: Batch size for evaluation
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 for greedy)
        """
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Create reward function
        reward_config = reward_config or {}
        self.reward_fn = TaskRewardFunction(
            correct_reward=reward_config.get("correct_reward", 1.0),
            incorrect_reward=reward_config.get("incorrect_reward", 0.0),
            math_tolerance=reward_config.get("math_tolerance", 1e-4),
        )

    def evaluate(
        self,
        model: Any,
        tokenizer: Any,
        samples: list[UltraInteractSample],
        model_path: str = "unknown",
        show_progress: bool = True,
    ) -> RLEvaluationResult:
        """Evaluate a model on UltraInteract samples.

        Args:
            model: HuggingFace model to evaluate
            tokenizer: Tokenizer for the model
            samples: List of samples to evaluate
            model_path: Path to model for logging
            show_progress: Whether to show progress bar

        Returns:
            RLEvaluationResult
        """
        import time

        start_time = time.time()

        # Ensure model is in eval mode
        model.eval()

        # Group samples by task type for tracking
        by_task: dict[str, list[UltraInteractSample]] = defaultdict(list)
        for sample in samples:
            by_task[sample.task_type.value].append(sample)

        # Evaluate samples
        evaluation_samples: list[EvaluationSample] = []
        correct_by_task: dict[str, int] = defaultdict(int)
        total_by_task: dict[str, int] = defaultdict(int)

        iterator = tqdm(samples, desc="Evaluating") if show_progress else samples

        for sample in iterator:
            # Format prompt
            prompt = format_chat_prompt(sample, tokenizer)

            # Generate response
            response = self._generate(model, tokenizer, prompt)

            # Get expected answer
            expected = (
                sample.final_answer
                if sample.final_answer
                else sample._extract_answer_from_response()
            )

            # Compute reward
            reward_result = self.reward_fn(response, sample)

            # Track metrics
            task_type = sample.task_type.value
            total_by_task[task_type] += 1
            if reward_result.is_correct:
                correct_by_task[task_type] += 1

            # Store sample
            evaluation_samples.append(
                EvaluationSample(
                    sample_id=sample.id,
                    task_type=task_type,
                    prompt=prompt[:200],  # Truncate
                    response=response,
                    expected=expected,
                    is_correct=reward_result.is_correct,
                    reward=reward_result.reward,
                    feedback=reward_result.feedback,
                )
            )

        # Compute metrics
        total_correct = sum(correct_by_task.values())
        total_samples = sum(total_by_task.values())
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        per_task_accuracy = {
            task: correct_by_task[task] / total_by_task[task]
            if total_by_task[task] > 0
            else 0.0
            for task in total_by_task
        }

        wall_time = time.time() - start_time

        return RLEvaluationResult(
            model_path=model_path,
            overall_accuracy=overall_accuracy,
            per_task_accuracy=per_task_accuracy,
            total_samples=total_samples,
            samples_by_task=dict(total_by_task),
            evaluation_samples=evaluation_samples,
            wall_time_seconds=wall_time,
            config={
                "batch_size": self.batch_size,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
            },
        )

    def _generate(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
    ) -> str:
        """Generate a response from the model.

        Args:
            model: HuggingFace model
            tokenizer: Tokenizer
            prompt: Input prompt

        Returns:
            Generated response string
        """
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )

        # Move to device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode only the generated tokens
        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        return response


def evaluate_model(
    model_path: Path | str,
    test_samples: list[UltraInteractSample] | None = None,
    test_size: int = 500,
    seed: int = 42,
    output_dir: Path | str | None = None,
    batch_size: int = 8,
    max_new_tokens: int = 512,
) -> RLEvaluationResult:
    """Evaluate a trained model on the UltraInteract test set.

    Args:
        model_path: Path to trained model
        test_samples: Optional pre-loaded test samples
        test_size: Number of test samples (if loading)
        seed: Random seed
        output_dir: Optional directory to save results
        batch_size: Batch size for evaluation
        max_new_tokens: Maximum tokens to generate

    Returns:
        RLEvaluationResult
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(model_path)

    logger.info(f"Loading model from {model_path}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Load test samples if not provided
    if test_samples is None:
        logger.info(f"Loading {test_size} test samples")
        loader = UltraInteractLoader(seed=seed)
        splits = loader.load_stratified(
            train_size=0,
            val_size=0,
            test_size=test_size,
        )
        test_samples = splits["test"]

    # Evaluate
    evaluator = RLModelEvaluator(
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )

    result = evaluator.evaluate(
        model=model,
        tokenizer=tokenizer,
        samples=test_samples,
        model_path=str(model_path),
    )

    # Save if output dir provided
    if output_dir:
        result.save(Path(output_dir))

    return result


def compare_models(
    base_model: str,
    trained_model: Path | str,
    test_samples: list[UltraInteractSample] | None = None,
    test_size: int = 500,
    seed: int = 42,
    output_dir: Path | str | None = None,
    batch_size: int = 8,
    max_new_tokens: int = 512,
) -> ModelComparisonResult:
    """Compare a trained model against a base model.

    Args:
        base_model: HuggingFace model ID for base model
        trained_model: Path to trained model
        test_samples: Optional pre-loaded test samples
        test_size: Number of test samples
        seed: Random seed
        output_dir: Optional directory to save results
        batch_size: Batch size for evaluation
        max_new_tokens: Maximum tokens to generate

    Returns:
        ModelComparisonResult
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    trained_model = Path(trained_model)

    # Load test samples if not provided
    if test_samples is None:
        logger.info(f"Loading {test_size} test samples")
        loader = UltraInteractLoader(seed=seed)
        splits = loader.load_stratified(
            train_size=0,
            val_size=0,
            test_size=test_size,
        )
        test_samples = splits["test"]

    evaluator = RLModelEvaluator(
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )

    # Evaluate base model
    logger.info(f"Evaluating base model: {base_model}")
    base_tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
    )
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token

    base_model_obj = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    base_result = evaluator.evaluate(
        model=base_model_obj,
        tokenizer=base_tokenizer,
        samples=test_samples,
        model_path=base_model,
    )

    # Free memory
    del base_model_obj
    torch.cuda.empty_cache()

    # Evaluate trained model
    logger.info(f"Evaluating trained model: {trained_model}")
    trained_tokenizer = AutoTokenizer.from_pretrained(
        trained_model,
        trust_remote_code=True,
    )
    if trained_tokenizer.pad_token is None:
        trained_tokenizer.pad_token = trained_tokenizer.eos_token

    trained_model_obj = AutoModelForCausalLM.from_pretrained(
        trained_model,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    trained_result = evaluator.evaluate(
        model=trained_model_obj,
        tokenizer=trained_tokenizer,
        samples=test_samples,
        model_path=str(trained_model),
    )

    # Compute deltas
    accuracy_delta = trained_result.overall_accuracy - base_result.overall_accuracy

    all_tasks = set(base_result.per_task_accuracy.keys()) | set(
        trained_result.per_task_accuracy.keys()
    )
    per_task_delta = {
        task: trained_result.per_task_accuracy.get(task, 0.0)
        - base_result.per_task_accuracy.get(task, 0.0)
        for task in all_tasks
    }

    result = ModelComparisonResult(
        base_result=base_result,
        trained_result=trained_result,
        accuracy_delta=accuracy_delta,
        per_task_delta=per_task_delta,
    )

    # Save if output dir provided
    if output_dir:
        result.save(Path(output_dir))

    logger.info(
        f"Comparison complete: "
        f"Base accuracy={base_result.overall_accuracy:.1%}, "
        f"Trained accuracy={trained_result.overall_accuracy:.1%}, "
        f"Delta={accuracy_delta:+.1%}"
    )

    return result
