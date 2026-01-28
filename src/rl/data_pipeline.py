"""Data pipeline for GRPO training on UltraInteract.

This module provides:
- Dataset adapter for converting UltraInteract to TRL GRPO format
- Prompt formatting functions for each task type
- Custom Dataset class for TRL compatibility
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from datasets import Dataset

from src.data.ultrainteract import (
    UITaskType,
    UltraInteractLoader,
    UltraInteractSample,
)

logger = logging.getLogger(__name__)


# Task-specific prompt templates (fixed, not evolved)
TASK_TEMPLATES: dict[str, str] = {
    "Coding": """For coding problems:
- Write clean, well-documented code
- Handle edge cases explicitly
- Consider time and space complexity
- Use meaningful variable names
- Include comments for complex logic""",
    "Math_CoT": """For mathematical reasoning:
- Show all calculation steps explicitly
- State the mathematical principles used
- Verify intermediate results
- Present the final answer clearly (use \\boxed{} for LaTeX)""",
    "Math_PoT": """For program-of-thought math:
- Translate the mathematical problem into executable code
- Define helper functions for complex operations
- Print intermediate results for verification
- Compute and clearly output the final answer""",
    "Logic": """For logical reasoning:
- Identify all premises and constraints
- Make logical deductions explicit step by step
- Consider alternative interpretations
- State your conclusion clearly with supporting evidence""",
}

# System prompt for the model
SYSTEM_PROMPT = """You are a helpful assistant that solves problems step by step.
When solving problems, follow these reasoning principles:

1. Break complex problems into smaller, manageable steps
2. Verify each step before proceeding to the next
3. Consider edge cases and potential errors
4. State your assumptions explicitly
5. Show your work clearly at each stage

Apply these guidelines throughout your reasoning process."""


def format_prompt(
    sample: UltraInteractSample | dict[str, Any],
    include_system: bool = True,
    include_task_template: bool = True,
) -> str:
    """Format a sample into a prompt for the model.

    Args:
        sample: UltraInteract sample or dict
        include_system: Whether to include system prompt
        include_task_template: Whether to include task-specific template

    Returns:
        Formatted prompt string
    """
    # Extract fields
    if isinstance(sample, UltraInteractSample):
        instruction = sample.instruction
        task_type = sample.task_type.value
    else:
        instruction = sample.get("instruction") or sample.get("input", "")
        task_type = sample.get("task_type") or sample.get(
            "additional_context", {}
        ).get("task_type", "Coding")

    parts = []

    # Add system prompt
    if include_system:
        parts.append(SYSTEM_PROMPT)

    # Add task-specific template
    if include_task_template:
        template = TASK_TEMPLATES.get(task_type, "")
        if template:
            parts.append(template)

    # Add instruction
    parts.append(instruction)

    return "\n\n".join(parts)


def format_chat_prompt(
    sample: UltraInteractSample | dict[str, Any],
    tokenizer: Any,
    include_task_template: bool = True,
) -> str:
    """Format a sample as a chat prompt using the tokenizer's chat template.

    Args:
        sample: UltraInteract sample or dict
        tokenizer: HuggingFace tokenizer with chat template
        include_task_template: Whether to include task-specific template

    Returns:
        Formatted chat prompt string
    """
    # Extract fields
    if isinstance(sample, UltraInteractSample):
        instruction = sample.instruction
        task_type = sample.task_type.value
    else:
        instruction = sample.get("instruction") or sample.get("input", "")
        task_type = sample.get("task_type") or sample.get(
            "additional_context", {}
        ).get("task_type", "Coding")

    # Build system message
    system_parts = [SYSTEM_PROMPT]
    if include_task_template:
        template = TASK_TEMPLATES.get(task_type, "")
        if template:
            system_parts.append(template)
    system_message = "\n\n".join(system_parts)

    # Create chat messages
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": instruction},
    ]

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return prompt


@dataclass
class GRPOSample:
    """A sample formatted for TRL GRPO training.

    Attributes:
        prompt: The formatted prompt
        ground_truth: Expected answer for reward computation
        task_type: Task type for task-specific grading
        sample_id: Original sample ID
        metadata: Additional metadata
    """

    prompt: str
    ground_truth: str
    task_type: str
    sample_id: str
    metadata: dict[str, Any] | None = None


class UltraInteractGRPODataset:
    """Dataset adapter for TRL GRPO training.

    Converts UltraInteract samples to the format expected by TRL's GRPOTrainer.
    """

    def __init__(
        self,
        samples: list[UltraInteractSample],
        tokenizer: Any | None = None,
        use_chat_template: bool = True,
        include_task_template: bool = True,
    ) -> None:
        """Initialize the dataset.

        Args:
            samples: List of UltraInteract samples
            tokenizer: Optional tokenizer for chat template formatting
            use_chat_template: Whether to use tokenizer's chat template
            include_task_template: Whether to include task-specific templates
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.use_chat_template = use_chat_template and tokenizer is not None
        self.include_task_template = include_task_template

        # Pre-format samples
        self._formatted_samples = self._format_all()

    def _format_all(self) -> list[GRPOSample]:
        """Format all samples."""
        formatted = []
        for sample in self.samples:
            if self.use_chat_template:
                prompt = format_chat_prompt(
                    sample,
                    self.tokenizer,
                    include_task_template=self.include_task_template,
                )
            else:
                prompt = format_prompt(
                    sample,
                    include_system=True,
                    include_task_template=self.include_task_template,
                )

            ground_truth = (
                sample.final_answer
                if sample.final_answer
                else sample._extract_answer_from_response()
            )

            formatted.append(
                GRPOSample(
                    prompt=prompt,
                    ground_truth=ground_truth,
                    task_type=sample.task_type.value,
                    sample_id=sample.id,
                    metadata={
                        "dataset_source": sample.dataset_source,
                        "has_code": sample.code_block is not None,
                    },
                )
            )
        return formatted

    def __len__(self) -> int:
        return len(self._formatted_samples)

    def __getitem__(self, idx: int) -> GRPOSample:
        return self._formatted_samples[idx]

    def to_hf_dataset(self) -> Dataset:
        """Convert to HuggingFace Dataset for TRL.

        Returns:
            HuggingFace Dataset with 'prompt', 'ground_truth', 'task_type' columns
        """
        return Dataset.from_dict(
            {
                "prompt": [s.prompt for s in self._formatted_samples],
                "ground_truth": [s.ground_truth for s in self._formatted_samples],
                "task_type": [s.task_type for s in self._formatted_samples],
                "sample_id": [s.sample_id for s in self._formatted_samples],
            }
        )


def create_grpo_dataset(
    samples: list[UltraInteractSample],
    tokenizer: Any | None = None,
    use_chat_template: bool = True,
) -> Dataset:
    """Create a HuggingFace Dataset for TRL GRPO training.

    Args:
        samples: List of UltraInteract samples
        tokenizer: Optional tokenizer for chat template formatting
        use_chat_template: Whether to use tokenizer's chat template

    Returns:
        HuggingFace Dataset ready for GRPOTrainer
    """
    dataset = UltraInteractGRPODataset(
        samples=samples,
        tokenizer=tokenizer,
        use_chat_template=use_chat_template,
    )
    return dataset.to_hf_dataset()


def load_ultrainteract_for_grpo(
    train_size: int = 5000,
    val_size: int = 500,
    tokenizer: Any | None = None,
    seed: int = 42,
    task_types: list[str] | None = None,
) -> dict[str, Dataset]:
    """Load UltraInteract dataset formatted for GRPO training.

    Args:
        train_size: Number of training samples
        val_size: Number of validation samples
        tokenizer: Optional tokenizer for chat template formatting
        seed: Random seed
        task_types: Optional list of task types to include

    Returns:
        Dictionary with 'train' and 'val' HuggingFace Datasets
    """
    loader = UltraInteractLoader(seed=seed)

    # Convert task types to enums
    task_type_enums = None
    if task_types:
        task_type_enums = [UITaskType.from_string(t) for t in task_types]

    # Load stratified splits
    splits = loader.load_stratified(
        train_size=train_size,
        val_size=val_size,
        test_size=0,
        task_types=task_type_enums,
    )

    logger.info(f"Loaded {len(splits['train'])} train, {len(splits['val'])} val samples")

    # Log task distribution
    train_dist = loader.get_task_distribution(splits["train"])
    logger.info(f"Train task distribution: {train_dist}")

    # Convert to GRPO format
    train_dataset = create_grpo_dataset(splits["train"], tokenizer=tokenizer)
    val_dataset = create_grpo_dataset(splits["val"], tokenizer=tokenizer)

    return {"train": train_dataset, "val": val_dataset}


def collate_grpo_batch(
    batch: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int = 2048,
) -> dict[str, Any]:
    """Collate a batch of GRPO samples for training.

    Args:
        batch: List of sample dictionaries
        tokenizer: Tokenizer for encoding
        max_length: Maximum sequence length

    Returns:
        Collated batch dictionary with tokenized inputs
    """
    prompts = [sample["prompt"] for sample in batch]
    ground_truths = [sample["ground_truth"] for sample in batch]
    task_types = [sample["task_type"] for sample in batch]

    # Tokenize prompts
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "ground_truth": ground_truths,
        "task_type": task_types,
    }
