"""UltraInteract dataset loader for LORE experiments.

UltraInteract is a large-scale reasoning dataset with step-by-step solutions
across multiple task types: Coding, Math_CoT, Math_PoT, and Logic.

Dataset source: openbmb/UltraInteract_sft
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from datasets import load_dataset as hf_load_dataset


class UITaskType(Enum):
    """UltraInteract task types."""

    CODING = "Coding"
    MATH_COT = "Math_CoT"
    MATH_POT = "Math_PoT"
    LOGIC = "Logic"

    @classmethod
    def from_string(cls, value: str) -> UITaskType:
        """Convert string to UITaskType."""
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"Unknown task type: {value}")


@dataclass
class UltraInteractSample:
    """A sample from the UltraInteract dataset.

    Attributes:
        id: Unique identifier (e.g., "92a7356beaa75391539cb79624bdf85b7e5d36d5e74e46a2f1c4001f40b3e055_1_1")
        parent_id: Parent ID for tracking related samples
        task_type: Type of task (Coding, Math_CoT, Math_PoT, Logic)
        dataset_source: Source dataset (codecontest, MATH, hotpotqa, etc.)
        instruction: The task instruction/question
        response: Full step-by-step response
        reasoning_steps: Parsed reasoning steps (Step 1: ..., Step 2: ...)
        final_answer: Extracted final answer if available
        code_block: Extracted code block if present
        metadata: Additional metadata
    """

    id: str
    parent_id: str
    task_type: UITaskType
    dataset_source: str
    instruction: str
    response: str
    reasoning_steps: list[str] = field(default_factory=list)
    final_answer: str | None = None
    code_block: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_gepa_format(self) -> dict[str, Any]:
        """Convert to GEPA evaluation format.

        Returns:
            Dictionary with fields expected by GEPA adapter.
        """
        return {
            "id": self.id,
            "input": self.instruction,
            "answer": self.final_answer or self._extract_answer_from_response(),
            "additional_context": {
                "task_type": self.task_type.value,
                "dataset_source": self.dataset_source,
                "reasoning_target": "\n".join(self.reasoning_steps),
                "has_code": self.code_block is not None,
            },
        }

    def _extract_answer_from_response(self) -> str:
        """Extract answer from response if final_answer not set."""
        # For coding tasks, the code block is the answer
        if self.task_type == UITaskType.CODING and self.code_block:
            return self.code_block

        # For Math_PoT, try to find a numeric answer, not the code
        if self.task_type == UITaskType.MATH_POT:
            # Try common answer patterns first
            patterns = [
                r"\\boxed\{(.+?)\}",
                r"(?:The\s+)?answer\s+is[:\s]+([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
                r"print\s*\(\s*([+-]?\d+(?:\.\d+)?)\s*\)",
            ]
            for pattern in patterns:
                match = re.search(pattern, self.response, re.IGNORECASE | re.MULTILINE)
                if match:
                    return match.group(1).strip().replace(",", "")

            # If we have code, try to find the last number in it
            if self.code_block:
                numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", self.code_block)
                if numbers:
                    return numbers[-1]

            # Last resort: use code block (will likely fail evaluation, but better than nothing)
            if self.code_block:
                return self.code_block

        # Try common answer patterns
        patterns = [
            r"Answer:\s*(.+?)$",
            r"The answer is\s*(.+?)$",
            r"Therefore,?\s*the answer is\s*(.+?)$",
            r"\\boxed\{(.+?)\}",
        ]

        for pattern in patterns:
            match = re.search(pattern, self.response, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()

        # Fallback: return last reasoning step or full response
        if self.reasoning_steps:
            return self.reasoning_steps[-1]
        return self.response

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "task_type": self.task_type.value,
            "dataset_source": self.dataset_source,
            "instruction": self.instruction,
            "response": self.response,
            "reasoning_steps": self.reasoning_steps,
            "final_answer": self.final_answer,
            "code_block": self.code_block,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UltraInteractSample:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            parent_id=data["parent_id"],
            task_type=UITaskType.from_string(data["task_type"]),
            dataset_source=data["dataset_source"],
            instruction=data["instruction"],
            response=data["response"],
            reasoning_steps=data.get("reasoning_steps", []),
            final_answer=data.get("final_answer"),
            code_block=data.get("code_block"),
            metadata=data.get("metadata", {}),
        )


class UltraInteractLoader:
    """Loader for UltraInteract dataset with task-type stratification.

    Example usage:
        loader = UltraInteractLoader(seed=42)

        # Load all samples
        samples = loader.load(sample_size=1000)

        # Load specific task types
        coding_samples = loader.load(task_types=[UITaskType.CODING])

        # Load with stratified splits
        splits = loader.load_stratified(train_size=1000, val_size=200)
    """

    DATASET_SOURCE = "openbmb/UltraInteract_sft"

    def __init__(
        self,
        seed: int = 42,
        cache_dir: Path | None = None,
    ) -> None:
        """Initialize the loader.

        Args:
            seed: Random seed for sampling
            cache_dir: Optional cache directory
        """
        self.seed = seed
        self.cache_dir = cache_dir
        random.seed(seed)

    def load(
        self,
        split: str = "train",
        sample_size: int | None = None,
        task_types: list[UITaskType] | None = None,
    ) -> list[UltraInteractSample]:
        """Load UltraInteract samples.

        Args:
            split: Dataset split (default: "train")
            sample_size: Optional maximum number of samples
            task_types: Optional list of task types to filter

        Returns:
            List of UltraInteractSample objects
        """
        # Load from HuggingFace
        dataset = hf_load_dataset(
            self.DATASET_SOURCE,
            split=split,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
        )

        # Convert to samples
        samples = []
        for item in dataset:
            try:
                sample = self._convert_item(item)

                # Filter by task type if specified
                if task_types and sample.task_type not in task_types:
                    continue

                samples.append(sample)
            except Exception as e:
                # Skip malformed items
                continue

        # Shuffle and sample
        random.shuffle(samples)
        if sample_size and len(samples) > sample_size:
            samples = samples[:sample_size]

        return samples

    def load_stratified(
        self,
        train_size: int = 1000,
        val_size: int = 200,
        test_size: int = 200,
        task_types: list[UITaskType] | None = None,
    ) -> dict[str, list[UltraInteractSample]]:
        """Load with stratified train/val/test splits.

        Maintains task type distribution across splits.

        Args:
            train_size: Number of training samples
            val_size: Number of validation samples
            test_size: Number of test samples
            task_types: Optional list of task types to include

        Returns:
            Dictionary with 'train', 'val', 'test' keys
        """
        # Load all samples - use larger buffer since Logic is rare (~4%)
        # We need enough samples to ensure all task types are represented
        total_needed = train_size + val_size + test_size
        # Logic is ~4% of dataset, so we need ~25x more samples to get enough
        # Use buffer of 5x minimum to ensure adequate representation
        buffer_factor = max(5.0, 25.0 * (total_needed / 1000))
        all_samples = self.load(
            sample_size=int(total_needed * buffer_factor),
            task_types=task_types,
        )

        # Group by task type
        by_task: dict[UITaskType, list[UltraInteractSample]] = defaultdict(list)
        for sample in all_samples:
            by_task[sample.task_type].append(sample)

        # Calculate per-task sizes
        active_tasks = task_types or list(UITaskType)
        n_tasks = len(active_tasks)

        splits: dict[str, list[UltraInteractSample]] = {
            "train": [],
            "val": [],
            "test": [],
        }

        # Stratified split per task type
        for task_type in active_tasks:
            task_samples = by_task.get(task_type, [])
            if not task_samples:
                continue

            random.shuffle(task_samples)

            # Calculate proportional sizes for this task
            task_train = train_size // n_tasks
            task_val = val_size // n_tasks
            task_test = test_size // n_tasks
            total_needed = task_train + task_val + task_test

            # If we don't have enough samples, split proportionally
            if len(task_samples) < total_needed:
                n = len(task_samples)
                # Split proportionally: train gets ~75%, val gets ~20%, test gets ~5%
                train_ratio = task_train / total_needed if total_needed > 0 else 0.75
                val_ratio = task_val / total_needed if total_needed > 0 else 0.20
                task_train = max(1, int(n * train_ratio))
                task_val = max(1, int(n * val_ratio))
                task_test = max(0, n - task_train - task_val)

            # Assign to splits
            idx = 0
            splits["train"].extend(task_samples[idx : idx + task_train])
            idx += task_train
            splits["val"].extend(task_samples[idx : idx + task_val])
            idx += task_val
            splits["test"].extend(task_samples[idx : idx + task_test])

        # Shuffle each split
        for split_name in splits:
            random.shuffle(splits[split_name])

        return splits

    def _convert_item(self, item: dict[str, Any]) -> UltraInteractSample:
        """Convert raw HuggingFace item to UltraInteractSample.

        Args:
            item: Raw item from HuggingFace dataset

        Returns:
            UltraInteractSample object
        """
        # Parse task type
        task_type = UITaskType.from_string(item["task"])

        # Parse response into components
        response = item["response"]
        reasoning_steps, final_answer, code_block = self._parse_response(
            response, task_type
        )

        return UltraInteractSample(
            id=item["id"],
            parent_id=item["parent_id"],
            task_type=task_type,
            dataset_source=item.get("dataset", "unknown"),
            instruction=item["instruction"],
            response=response,
            reasoning_steps=reasoning_steps,
            final_answer=final_answer,
            code_block=code_block,
            metadata={
                "original_item": {
                    "id": item["id"],
                    "task": item["task"],
                    "dataset": item.get("dataset"),
                }
            },
        )

    def _parse_response(
        self,
        response: str,
        task_type: UITaskType,
    ) -> tuple[list[str], str | None, str | None]:
        """Parse response into reasoning steps, final answer, and code block.

        Args:
            response: Raw response string
            task_type: Task type for context-aware parsing

        Returns:
            Tuple of (reasoning_steps, final_answer, code_block)
        """
        reasoning_steps = []
        final_answer = None
        code_block = None

        # Extract numbered steps (Step 1: ..., Step 2: ...)
        # Handle both "Step 1:" and "# Step 1:" patterns
        step_pattern = r"(?:#\s*)?Step\s+(\d+):\s*(.+?)(?=(?:#\s*)?Step\s+\d+:|```|$)"
        matches = re.findall(step_pattern, response, re.DOTALL | re.IGNORECASE)
        for num, content in matches:
            step_text = content.strip()
            if step_text:
                reasoning_steps.append(f"Step {num}: {step_text}")

        # Extract code block (```python ... ```)
        code_pattern = r"```(?:python)?\s*\n?(.*?)```"
        code_match = re.search(code_pattern, response, re.DOTALL)
        if code_match:
            code_block = code_match.group(1).strip()

        # Extract final answer based on task type
        if task_type == UITaskType.MATH_COT:
            # Look for boxed answer or explicit answer statement
            boxed_match = re.search(r"\\boxed\{(.+?)\}", response)
            if boxed_match:
                final_answer = boxed_match.group(1).strip()
            else:
                # Try "Answer: X" or "answer is X" patterns
                answer_match = re.search(
                    r"(?:The\s+)?answer\s+is[:\s]+([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
                    response,
                    re.IGNORECASE,
                )
                if answer_match:
                    final_answer = answer_match.group(1).strip().replace(",", "")

        elif task_type == UITaskType.MATH_POT:
            # For Math_PoT, extract the NUMERIC result, not the code
            # Priority order: boxed > explicit print(number) > text answer > code analysis

            # 1. Check for boxed answer
            boxed_match = re.search(r"\\boxed\{(.+?)\}", response)
            if boxed_match:
                final_answer = boxed_match.group(1).strip()

            # 2. Look for print() with a literal number
            if not final_answer:
                print_match = re.search(
                    r"print\s*\(\s*([+-]?\d+(?:\.\d+)?)\s*\)",
                    response,
                )
                if print_match:
                    final_answer = print_match.group(1).strip()

            # 3. Look for "The answer is X" pattern in text
            if not final_answer:
                answer_match = re.search(
                    r"(?:The\s+)?answer\s+is[:\s]+([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
                    response,
                    re.IGNORECASE,
                )
                if answer_match:
                    final_answer = answer_match.group(1).strip().replace(",", "")

            # 4. Try to extract from code patterns
            if not final_answer and code_block:
                # Look for variable assignment followed by print
                # e.g., "result = 42\nprint(result)" -> 42
                var_assignments = dict(re.findall(
                    r"(\w+)\s*=\s*([+-]?\d+(?:\.\d+)?)\s*$",
                    code_block,
                    re.MULTILINE
                ))

                # Find what's being printed
                printed_vars = re.findall(r"print\s*\(\s*(\w+)\s*\)", code_block)
                for var in printed_vars:
                    if var in var_assignments:
                        final_answer = var_assignments[var]
                        break

                # If still no answer, look for last numeric assignment
                if not final_answer:
                    last_num = re.findall(
                        r"=\s*([+-]?\d+(?:\.\d+)?)\s*$",
                        code_block,
                        re.MULTILINE
                    )
                    if last_num:
                        final_answer = last_num[-1]

        elif task_type == UITaskType.LOGIC:
            # Look for various answer patterns in logic tasks
            # Try "Answer: X" pattern
            answer_match = re.search(
                r"Answer:\s*(.+?)$",
                response,
                re.IGNORECASE | re.MULTILINE,
            )
            if answer_match:
                final_answer = answer_match.group(1).strip()
            else:
                # Try "The answer is X" pattern
                answer_match = re.search(
                    r"(?:The\s+)?answer\s+is[:\s]+(.+?)(?:\.|$)",
                    response,
                    re.IGNORECASE,
                )
                if answer_match:
                    final_answer = answer_match.group(1).strip()

        elif task_type == UITaskType.CODING:
            # For coding, extract actual Python code from code block
            # UltraInteract code blocks contain real code with "# Step N:" comments
            if code_block:
                # Code block exists - it should contain executable Python
                final_answer = code_block
            else:
                # No code block found, try to extract code from response
                # Look for Python code patterns
                code_patterns = [
                    r'(def\s+\w+\s*\([^)]*\)[\s\S]*?)(?=\n\S|$)',  # Function def
                    r'(class\s+\w+[\s\S]*?)(?=\nclass\s|\ndef\s|$)',  # Class def
                ]
                for pattern in code_patterns:
                    match = re.search(pattern, response)
                    if match:
                        final_answer = match.group(1).strip()
                        break

        return reasoning_steps, final_answer, code_block

    def get_task_distribution(
        self,
        samples: list[UltraInteractSample],
    ) -> dict[str, int]:
        """Get distribution of task types in samples.

        Args:
            samples: List of samples

        Returns:
            Dictionary mapping task type to count
        """
        distribution: dict[str, int] = defaultdict(int)
        for sample in samples:
            distribution[sample.task_type.value] += 1
        return dict(distribution)


def create_gepa_dataset(
    samples: list[UltraInteractSample],
) -> list[dict[str, Any]]:
    """Convert UltraInteract samples to GEPA format.

    Args:
        samples: List of UltraInteractSample objects

    Returns:
        List of dictionaries in GEPA format
    """
    return [s.to_gepa_format() for s in samples]


def load_ultrainteract(
    sample_size: int | None = None,
    task_types: list[str] | None = None,
    seed: int = 42,
) -> list[UltraInteractSample]:
    """Convenience function to load UltraInteract.

    Args:
        sample_size: Optional maximum number of samples
        task_types: Optional list of task type strings
        seed: Random seed

    Returns:
        List of UltraInteractSample objects
    """
    loader = UltraInteractLoader(seed=seed)

    # Convert task type strings to enums
    task_type_enums = None
    if task_types:
        task_type_enums = [UITaskType.from_string(t) for t in task_types]

    return loader.load(sample_size=sample_size, task_types=task_type_enums)
