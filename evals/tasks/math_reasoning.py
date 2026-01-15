"""Math reasoning evaluation task using inspect_ai.

Evaluates mathematical problem-solving capabilities using GSM8K and MATH datasets.
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import generate, system_message

from evals.scorers.math_scorer import math_answer_scorer


def gsm8k_record_to_sample(record: dict) -> Sample:
    """Convert GSM8K record to inspect_ai Sample."""
    return Sample(
        input=record["question"],
        target=record["answer"].split("####")[-1].strip(),
        metadata={
            "full_answer": record["answer"],
        },
    )


def math_record_to_sample(record: dict) -> Sample:
    """Convert MATH dataset record to inspect_ai Sample."""
    import re

    # Extract boxed answer
    solution = record["solution"]
    boxed_match = re.search(r"\\boxed{(.*)}", solution)
    target = boxed_match.group(1) if boxed_match else solution.split()[-1]

    return Sample(
        input=record["problem"],
        target=target,
        metadata={
            "level": record.get("level"),
            "type": record.get("type"),
            "full_solution": solution,
        },
    )


@task
def math_reasoning_task(
    dataset: str = "gsm8k",
    system_prompt: str | None = None,
    optimized_prompt: str | None = None,
    sample_limit: int | None = None,
) -> Task:
    """Create math reasoning evaluation task.

    Args:
        dataset: Dataset to use ("gsm8k" or "math")
        system_prompt: Optional system prompt
        optimized_prompt: Optional optimized reasoning prompt to prepend
        sample_limit: Limit number of samples

    Returns:
        inspect_ai Task
    """
    if dataset == "gsm8k":
        ds = hf_dataset(
            "openai/gsm8k",
            split="test",
            sample_fields=gsm8k_record_to_sample,
            limit=sample_limit,
        )
    elif dataset == "math":
        ds = hf_dataset(
            "lighteval/MATH",
            split="test",
            sample_fields=math_record_to_sample,
            limit=sample_limit,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Build solver pipeline
    solvers = []

    if system_prompt:
        solvers.append(system_message(system_prompt))

    # If optimized prompt provided, prepend to user message
    if optimized_prompt:
        # This would be handled via prompt template in practice
        pass

    solvers.append(generate())

    return Task(
        dataset=ds,
        solver=solvers,
        scorer=math_answer_scorer(),
        config={
            "dataset": dataset,
            "optimized_prompt": optimized_prompt,
        },
    )


@task
def gsm8k_test(sample_limit: int | None = None) -> Task:
    """GSM8K test set evaluation."""
    return math_reasoning_task(dataset="gsm8k", sample_limit=sample_limit)


@task
def math_test(sample_limit: int | None = None) -> Task:
    """MATH test set evaluation."""
    return math_reasoning_task(dataset="math", sample_limit=sample_limit)
