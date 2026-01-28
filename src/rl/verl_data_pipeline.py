"""Data pipeline for Verl GRPO training with open-r1 and SynLogic datasets.

This module provides:
- Dataset loaders for OpenR1-Math-220k and SynLogic datasets
- Prompt formatting functions for Verl's dataflow
- Unified dataset class for multi-task training
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from datasets import Dataset, concatenate_datasets
from datasets import load_dataset as hf_load_dataset

logger = logging.getLogger(__name__)


# Task type enumeration for Verl training
class VerlTaskType:
    """Task types supported by Verl training pipeline."""
    MATH = "math"
    LOGIC = "logic"
    CODE = "code"


@dataclass
class VerlSample:
    """A sample formatted for Verl GRPO training.

    Attributes:
        prompt: The formatted prompt (chat messages or string)
        answer: Expected answer for reward computation
        task_type: Task type for task-specific grading
        sample_id: Original sample ID
        extra_info: Additional task-specific information
        metadata: Additional metadata
    """

    prompt: str | list[dict[str, str]]
    answer: str
    task_type: str
    sample_id: str
    extra_info: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def load_openr1_math(
    split: str = "train",
    sample_size: int | None = None,
    seed: int = 42,
    subset: str | None = None,
) -> list[VerlSample]:
    """Load OpenR1-Math-220k dataset.

    The dataset contains math problems with verified reasoning traces.
    Source: https://huggingface.co/datasets/open-r1/OpenR1-Math-220k

    Args:
        split: Dataset split to load
        sample_size: Number of samples to load (None for all)
        seed: Random seed for sampling
        subset: Optional subset name (e.g., "default", "extended")

    Returns:
        List of VerlSample objects
    """
    logger.info(f"Loading OpenR1-Math-220k dataset (split={split})")

    try:
        if subset:
            dataset = hf_load_dataset("open-r1/OpenR1-Math-220k", subset, split=split)
        else:
            dataset = hf_load_dataset("open-r1/OpenR1-Math-220k", split=split)
    except Exception as e:
        logger.warning(f"Failed to load open-r1/OpenR1-Math-220k: {e}")
        logger.info("Attempting to load alternative math dataset...")
        # Fallback to another math dataset if open-r1 is not available
        dataset = hf_load_dataset("lighteval/MATH", "all", split="train")

    # Sample if needed
    if sample_size and len(dataset) > sample_size:
        dataset = dataset.shuffle(seed=seed).select(range(sample_size))

    samples = []
    for i, item in enumerate(dataset):
        # OpenR1-Math-220k schema:
        # - problem: the question text
        # - answer: the final answer (use this for reward)
        # - solution: full solution explanation
        # - messages: chat format [{"role": "user", "content": ...}]

        # Extract question - prefer messages for chat format
        if "messages" in item and isinstance(item["messages"], list) and len(item["messages"]) > 0:
            # Use chat format from messages
            question = item["messages"]
        elif "problem" in item:
            question = item["problem"]
        elif "question" in item:
            question = item["question"]
        elif "prompt" in item:
            if isinstance(item["prompt"], list):
                question = item["prompt"][-1].get("content", str(item["prompt"]))
            else:
                question = item["prompt"]
        else:
            question = str(item.get("input", ""))

        # Extract answer - prefer 'answer' field for OpenR1-Math
        # Note: 'solution' is the full solution text, 'answer' is the final answer
        if "answer" in item:
            answer = item["answer"]
        elif "solution" in item:
            answer = item["solution"]
        elif "reward_model" in item:
            answer = item["reward_model"].get("answer", "")
        else:
            answer = ""

        # Extract boxed answer if present in the answer
        if isinstance(answer, str):
            boxed_match = re.search(r"\\boxed\{([^}]+)\}", answer)
            if boxed_match:
                answer = boxed_match.group(1)

        samples.append(
            VerlSample(
                prompt=question,
                answer=str(answer),
                task_type=VerlTaskType.MATH,
                sample_id=f"openr1_math_{i}",
                extra_info={
                    "solution": item.get("solution", ""),
                    "problem_type": item.get("problem_type", ""),
                },
                metadata={
                    "source": "open-r1/OpenR1-Math-220k",
                    "split": split,
                    "original_index": i,
                },
            )
        )

    logger.info(f"Loaded {len(samples)} samples from OpenR1-Math-220k")
    return samples


def load_synlogic(
    difficulty: str = "easy",
    split: str = "train",
    sample_size: int | None = None,
    seed: int = 42,
    task_names: list[str] | None = None,
) -> list[VerlSample]:
    """Load SynLogic dataset for logic reasoning.

    The dataset contains 35 verifiable logic tasks with rule-based verifiers.
    Source: https://huggingface.co/datasets/MiniMaxAI/SynLogic

    Args:
        difficulty: Difficulty level ("easy" or "hard")
        split: Dataset split to load
        sample_size: Number of samples to load (None for all)
        seed: Random seed for sampling
        task_names: Optional list of specific task names to include

    Returns:
        List of VerlSample objects
    """
    logger.info(f"Loading SynLogic dataset (difficulty={difficulty}, split={split})")

    try:
        dataset = hf_load_dataset("MiniMaxAI/SynLogic", difficulty, split=split)
    except Exception as e:
        logger.warning(f"Failed to load MiniMaxAI/SynLogic: {e}")
        # Return empty list if dataset is not available
        return []

    # Filter by data_source if specified (SynLogic uses data_source, not task_name)
    if task_names:
        task_names_set = set(task_names)
        dataset = dataset.filter(
            lambda x: x.get("data_source", "") in task_names_set
        )

    # Sample if needed
    if sample_size and len(dataset) > sample_size:
        dataset = dataset.shuffle(seed=seed).select(range(sample_size))

    samples = []
    for i, item in enumerate(dataset):
        # SynLogic uses chat format with prompt as list of messages
        prompt = item.get("prompt", [])
        if isinstance(prompt, list) and len(prompt) > 0:
            # Extract the user message content
            user_content = prompt[-1].get("content", "") if isinstance(prompt[-1], dict) else str(prompt[-1])
        else:
            user_content = str(prompt)

        # Extract answer from reward_model field
        reward_model = item.get("reward_model", {})
        answer = reward_model.get("answer", "")
        solution = reward_model.get("solution", "")

        # Get extra info for task-specific verification
        extra_info = {}
        if "extra_info" in item:
            extra_info_raw = item["extra_info"]
            if isinstance(extra_info_raw, dict):
                extra_info = extra_info_raw
            elif isinstance(extra_info_raw, str):
                try:
                    extra_info = json.loads(extra_info_raw)
                except json.JSONDecodeError:
                    extra_info = {"raw": extra_info_raw}

        # Include game_data_str if present
        if "game_data_str" in extra_info:
            pass  # Keep it for verification
        elif "game_data_str" in item:
            extra_info["game_data_str"] = item["game_data_str"]

        # SynLogic uses data_source field, not task_name
        task_name = item.get("data_source", "unknown")

        samples.append(
            VerlSample(
                prompt=prompt if isinstance(prompt, list) else user_content,
                answer=str(answer),
                task_type=VerlTaskType.LOGIC,
                sample_id=f"synlogic_{difficulty}_{i}",
                extra_info={
                    "solution": solution,
                    "task_name": task_name,
                    **extra_info,
                },
                metadata={
                    "source": "MiniMaxAI/SynLogic",
                    "difficulty": difficulty,
                    "split": split,
                    "original_index": i,
                },
            )
        )

    logger.info(f"Loaded {len(samples)} samples from SynLogic")
    return samples


def load_code_contests(
    split: str = "train",
    sample_size: int | None = None,
    seed: int = 42,
    difficulty: list[str] | None = None,
) -> list[VerlSample]:
    """Load code_contests dataset for code execution tasks.

    Source: https://huggingface.co/datasets/deepmind/code_contests

    Args:
        split: Dataset split to load
        sample_size: Number of samples to load (None for all)
        seed: Random seed for sampling
        difficulty: Optional list of difficulty levels to include

    Returns:
        List of VerlSample objects
    """
    logger.info(f"Loading code_contests dataset (split={split})")

    try:
        dataset = hf_load_dataset("deepmind/code_contests", split=split)
    except Exception as e:
        logger.warning(f"Failed to load deepmind/code_contests: {e}")
        return []

    # Filter by difficulty if specified
    if difficulty:
        difficulty_set = set(difficulty)
        dataset = dataset.filter(
            lambda x: x.get("difficulty", 0) in difficulty_set
        )

    # Sample if needed
    if sample_size and len(dataset) > sample_size:
        dataset = dataset.shuffle(seed=seed).select(range(sample_size))

    samples = []
    for i, item in enumerate(dataset):
        # Extract problem description
        description = item.get("description", "")

        # Extract test cases
        public_tests = item.get("public_tests", {})
        private_tests = item.get("private_tests", {})

        test_cases = []

        # Parse public tests
        if isinstance(public_tests, dict):
            inputs = public_tests.get("input", [])
            outputs = public_tests.get("output", [])
            for inp, out in zip(inputs, outputs):
                test_cases.append({"input": inp, "output": out, "type": "public"})

        # Parse private tests (for evaluation)
        if isinstance(private_tests, dict):
            inputs = private_tests.get("input", [])
            outputs = private_tests.get("output", [])
            for inp, out in zip(inputs, outputs):
                test_cases.append({"input": inp, "output": out, "type": "private"})

        # Get solutions if available (for reference)
        solutions = item.get("solutions", {})
        python_solutions = []
        if isinstance(solutions, dict):
            languages = solutions.get("language", [])
            solution_codes = solutions.get("solution", [])
            for lang, code in zip(languages, solution_codes):
                if lang in [3, "python", "python3", "PYTHON3"]:
                    python_solutions.append(code)

        samples.append(
            VerlSample(
                prompt=description,
                answer=python_solutions[0] if python_solutions else "",
                task_type=VerlTaskType.CODE,
                sample_id=f"code_contests_{i}",
                extra_info={
                    "test_cases": test_cases,
                    "difficulty": item.get("difficulty", 0),
                    "source_name": item.get("source", ""),
                },
                metadata={
                    "source": "deepmind/code_contests",
                    "split": split,
                    "original_index": i,
                    "has_solutions": len(python_solutions) > 0,
                },
            )
        )

    logger.info(f"Loaded {len(samples)} samples from code_contests")
    return samples


def format_prompt_for_verl(
    sample: VerlSample,
    tokenizer: Any = None,
    include_system_prompt: bool = True,
    max_length: int = 2048,
) -> str:
    """Format a sample prompt for Verl training.

    Args:
        sample: VerlSample object
        tokenizer: Optional tokenizer for chat template
        include_system_prompt: Whether to include system prompt
        max_length: Maximum prompt length

    Returns:
        Formatted prompt string
    """
    # System prompts per task type
    system_prompts = {
        VerlTaskType.MATH: (
            "You are a helpful math assistant. Solve problems step by step. "
            "Show your reasoning clearly and put your final answer in \\boxed{}."
        ),
        VerlTaskType.LOGIC: (
            "You are a helpful assistant solving logic puzzles. "
            "Think through the problem step by step inside <think>...</think> tags, "
            "then provide your final answer inside <answer>...</answer> tags."
        ),
        VerlTaskType.CODE: (
            "You are a helpful programming assistant. Write clean, efficient code "
            "that solves the given problem. Include proper input/output handling."
        ),
    }

    system_prompt = system_prompts.get(sample.task_type, "")

    # Handle chat format prompts
    if isinstance(sample.prompt, list):
        messages = []
        if include_system_prompt and system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(sample.prompt)

        if tokenizer and hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback: join messages
            parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append(f"[{role}]\n{content}")
            return "\n\n".join(parts)

    # Handle string prompts
    if tokenizer and hasattr(tokenizer, "apply_chat_template"):
        messages = []
        if include_system_prompt and system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": sample.prompt})

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Fallback: simple formatting
    if include_system_prompt and system_prompt:
        return f"{system_prompt}\n\n{sample.prompt}"
    return sample.prompt


def samples_to_hf_dataset(samples: list[VerlSample]) -> Dataset:
    """Convert VerlSamples to HuggingFace Dataset for Verl.

    Args:
        samples: List of VerlSample objects

    Returns:
        HuggingFace Dataset with columns for Verl training
    """
    return Dataset.from_dict(
        {
            "prompt": [s.prompt if isinstance(s.prompt, str) else s.prompt for s in samples],
            "answer": [s.answer for s in samples],
            "task_type": [s.task_type for s in samples],
            "sample_id": [s.sample_id for s in samples],
            "extra_info": [json.dumps(s.extra_info) for s in samples],
        }
    )


def load_combined_dataset(
    math_samples: int = 10000,
    logic_samples: int = 5000,
    code_samples: int = 2000,
    seed: int = 42,
    tokenizer: Any = None,
) -> Dataset:
    """Load and combine multiple datasets for multi-task Verl training.

    Args:
        math_samples: Number of math samples
        logic_samples: Number of logic samples
        code_samples: Number of code samples
        seed: Random seed
        tokenizer: Optional tokenizer for formatting

    Returns:
        Combined HuggingFace Dataset
    """
    all_samples = []

    # Load math samples
    if math_samples > 0:
        math_data = load_openr1_math(
            sample_size=math_samples,
            seed=seed,
        )
        all_samples.extend(math_data)

    # Load logic samples
    if logic_samples > 0:
        # Load from multiple difficulties
        easy_count = logic_samples // 2
        hard_count = logic_samples - easy_count

        easy_data = load_synlogic(
            difficulty="easy",
            sample_size=easy_count,
            seed=seed,
        )
        all_samples.extend(easy_data)

        hard_data = load_synlogic(
            difficulty="hard",
            sample_size=hard_count,
            seed=seed,
        )
        all_samples.extend(hard_data)

    # Load code samples
    if code_samples > 0:
        code_data = load_code_contests(
            sample_size=code_samples,
            seed=seed,
        )
        all_samples.extend(code_data)

    logger.info(
        f"Combined dataset: {len(all_samples)} samples "
        f"(math={math_samples}, logic={logic_samples}, code={code_samples})"
    )

    # Format prompts if tokenizer provided
    if tokenizer:
        for sample in all_samples:
            sample.prompt = format_prompt_for_verl(sample, tokenizer)

    return samples_to_hf_dataset(all_samples)


def load_verl_datasets(
    train_math: int = 10000,
    train_logic: int = 5000,
    train_code: int = 2000,
    val_math: int = 500,
    val_logic: int = 250,
    val_code: int = 100,
    seed: int = 42,
    tokenizer: Any = None,
) -> dict[str, Dataset]:
    """Load train and validation datasets for Verl GRPO.

    Args:
        train_math: Training math samples
        train_logic: Training logic samples
        train_code: Training code samples
        val_math: Validation math samples
        val_logic: Validation logic samples
        val_code: Validation code samples
        seed: Random seed
        tokenizer: Optional tokenizer for formatting

    Returns:
        Dictionary with 'train' and 'val' datasets
    """
    train_dataset = load_combined_dataset(
        math_samples=train_math,
        logic_samples=train_logic,
        code_samples=train_code,
        seed=seed,
        tokenizer=tokenizer,
    )

    val_dataset = load_combined_dataset(
        math_samples=val_math,
        logic_samples=val_logic,
        code_samples=val_code,
        seed=seed + 1,  # Different seed for validation
        tokenizer=tokenizer,
    )

    return {"train": train_dataset, "val": val_dataset}
