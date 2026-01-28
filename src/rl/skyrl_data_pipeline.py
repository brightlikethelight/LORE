"""Data pipeline for SkyRL GRPO training.

This module provides dataset adapters for SkyRL's PromptDataset format,
building on the existing Verl data pipeline components.

Key functions:
- convert_to_skyrl_format: Convert VerlSample to SkyRL format
- load_skyrl_datasets: Load train/val datasets for SkyRL training
"""

from __future__ import annotations

import json
import logging
from typing import Any

from datasets import Dataset

from src.rl.verl_data_pipeline import (
    VerlSample,
    VerlTaskType,
    load_openr1_math,
    load_synlogic,
    load_code_contests,
    format_prompt_for_verl,
)

logger = logging.getLogger(__name__)


def convert_to_skyrl_format(
    samples: list[VerlSample],
    tokenizer: Any = None,
) -> list[dict[str, Any]]:
    """Convert VerlSample list to SkyRL PromptDataset format.

    SkyRL expects dictionaries with:
    - prompt: The formatted prompt string or chat messages
    - metadata: Dictionary with answer, task_type, and other info

    Args:
        samples: List of VerlSample objects
        tokenizer: Optional tokenizer for formatting prompts

    Returns:
        List of dictionaries in SkyRL format
    """
    skyrl_data = []

    for sample in samples:
        # Format prompt if tokenizer provided
        if tokenizer:
            prompt = format_prompt_for_verl(sample, tokenizer)
        else:
            prompt = sample.prompt

        skyrl_data.append({
            "prompt": prompt,
            "metadata": {
                "answer": sample.answer,
                "task_type": sample.task_type,
                "sample_id": sample.sample_id,
                "extra_info": sample.extra_info,
            }
        })

    return skyrl_data


def samples_to_hf_dataset(
    samples: list[VerlSample],
    tokenizer: Any = None,
) -> Dataset:
    """Convert VerlSamples to HuggingFace Dataset for SkyRL.

    Args:
        samples: List of VerlSample objects
        tokenizer: Optional tokenizer for formatting

    Returns:
        HuggingFace Dataset compatible with SkyRL trainer
    """
    # Format prompts if tokenizer provided
    prompts = []
    for sample in samples:
        if tokenizer:
            prompt = format_prompt_for_verl(sample, tokenizer)
        else:
            prompt = sample.prompt if isinstance(sample.prompt, str) else json.dumps(sample.prompt)
        prompts.append(prompt)

    return Dataset.from_dict({
        "prompt": prompts,
        "answer": [s.answer for s in samples],
        "task_type": [s.task_type for s in samples],
        "sample_id": [s.sample_id for s in samples],
        "extra_info": [json.dumps(s.extra_info) for s in samples],
    })


def load_skyrl_datasets(
    math_samples: int = 10000,
    logic_samples: int = 5000,
    code_samples: int = 2000,
    val_ratio: float = 0.1,
    seed: int = 42,
    tokenizer: Any = None,
) -> dict[str, Dataset]:
    """Load train and validation datasets for SkyRL GRPO.

    This function provides a simpler interface than load_verl_datasets,
    automatically computing validation sizes based on a ratio.

    Args:
        math_samples: Total math samples (train + val)
        logic_samples: Total logic samples (train + val)
        code_samples: Total code samples (train + val)
        val_ratio: Ratio of samples to use for validation
        seed: Random seed
        tokenizer: Optional tokenizer for formatting

    Returns:
        Dictionary with 'train' and 'val' datasets
    """
    all_samples: list[VerlSample] = []

    # Load math samples
    if math_samples > 0:
        math_data = load_openr1_math(
            sample_size=math_samples,
            seed=seed,
        )
        all_samples.extend(math_data)
        logger.info(f"Loaded {len(math_data)} math samples")

    # Load logic samples (split between easy and hard)
    if logic_samples > 0:
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
        logger.info(f"Loaded {len(easy_data) + len(hard_data)} logic samples")

    # Load code samples
    if code_samples > 0:
        code_data = load_code_contests(
            sample_size=code_samples,
            seed=seed,
        )
        all_samples.extend(code_data)
        logger.info(f"Loaded {len(code_data)} code samples")

    # Shuffle and split
    import random
    random.seed(seed)
    random.shuffle(all_samples)

    val_size = int(len(all_samples) * val_ratio)
    train_samples = all_samples[val_size:]
    val_samples = all_samples[:val_size]

    logger.info(
        f"SkyRL datasets: {len(train_samples)} train, {len(val_samples)} val samples"
    )

    return {
        "train": samples_to_hf_dataset(train_samples, tokenizer),
        "val": samples_to_hf_dataset(val_samples, tokenizer),
    }


def load_smoke_test_datasets(
    math_samples: int = 50,
    logic_samples: int = 30,
    code_samples: int = 20,
    seed: int = 42,
    tokenizer: Any = None,
) -> dict[str, Dataset]:
    """Load small datasets for smoke testing SkyRL training.

    This provides a minimal dataset for quick validation that training
    runs without errors before committing to a full training run.

    Args:
        math_samples: Number of math samples
        logic_samples: Number of logic samples
        code_samples: Number of code samples
        seed: Random seed
        tokenizer: Optional tokenizer

    Returns:
        Dictionary with 'train' and 'val' datasets
    """
    return load_skyrl_datasets(
        math_samples=math_samples,
        logic_samples=logic_samples,
        code_samples=code_samples,
        val_ratio=0.2,  # Higher val ratio for smoke test
        seed=seed,
        tokenizer=tokenizer,
    )


class SkyRLPromptDataset:
    """Wrapper for SkyRL's PromptDataset interface.

    This class provides compatibility with SkyRL's expected dataset
    interface while using our VerlSample data model internally.
    """

    def __init__(
        self,
        samples: list[VerlSample] | list[dict[str, Any]],
        tokenizer: Any = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            samples: List of VerlSample or dict samples
            tokenizer: Optional tokenizer for formatting
        """
        self._tokenizer = tokenizer

        # Convert to internal format
        if samples and isinstance(samples[0], VerlSample):
            self._data = convert_to_skyrl_format(samples, tokenizer)  # type: ignore
        else:
            self._data = list(samples)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self._data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get item by index."""
        return self._data[idx]

    def __iter__(self):
        """Iterate over samples."""
        return iter(self._data)

    def get_prompt(self, idx: int) -> str:
        """Get prompt string for index."""
        item = self._data[idx]
        prompt = item.get("prompt", "")
        if isinstance(prompt, list):
            if self._tokenizer and hasattr(self._tokenizer, "apply_chat_template"):
                return self._tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            return "\n".join(str(m) for m in prompt)
        return prompt

    def get_metadata(self, idx: int) -> dict[str, Any]:
        """Get metadata for index."""
        return self._data[idx].get("metadata", {})

    def sample(self, n: int, seed: int | None = None) -> "SkyRLPromptDataset":
        """Sample n items from the dataset.

        Args:
            n: Number of items to sample
            seed: Optional random seed

        Returns:
            New SkyRLPromptDataset with sampled items
        """
        import random
        if seed is not None:
            random.seed(seed)

        indices = random.sample(range(len(self._data)), min(n, len(self._data)))
        sampled = [self._data[i] for i in indices]
        return SkyRLPromptDataset(sampled, self._tokenizer)

    def to_hf_dataset(self) -> Dataset:
        """Convert to HuggingFace Dataset."""
        return Dataset.from_dict({
            "prompt": [d.get("prompt", "") for d in self._data],
            "metadata": [json.dumps(d.get("metadata", {})) for d in self._data],
        })
