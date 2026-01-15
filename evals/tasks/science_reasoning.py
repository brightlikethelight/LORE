"""Science reasoning evaluation task using GPQA dataset.

Evaluates graduate-level science question answering capabilities.
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.solver import generate, multiple_choice, system_message

from evals.scorers.mcq_scorer import mcq_scorer


def gpqa_record_to_sample(record: dict) -> Sample:
    """Convert GPQA record to inspect_ai Sample."""
    # GPQA has question, correct answer, and distractor options
    question = record["Question"]
    correct = record["Correct Answer"]

    # Build choices list
    choices = [correct]
    for key in ["Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]:
        if key in record and record[key]:
            choices.append(record[key])

    # Shuffle choices and track correct index
    import random

    random.seed(hash(question))  # Deterministic shuffle
    random.shuffle(choices)
    correct_idx = choices.index(correct)
    correct_letter = chr(65 + correct_idx)  # A, B, C, D

    return Sample(
        input=question,
        choices=choices,
        target=correct_letter,
        metadata={
            "domain": record.get("High-level domain"),
            "subdomain": record.get("Subdomain"),
            "difficulty": record.get("Writer's Coverage Coverage"),
        },
    )


@task
def science_reasoning_task(
    subset: str = "gpqa_diamond",
    system_prompt: str | None = None,
    sample_limit: int | None = None,
) -> Task:
    """Create science reasoning evaluation task.

    Args:
        subset: GPQA subset ("gpqa_diamond", "gpqa_main", "gpqa_extended")
        system_prompt: Optional system prompt
        sample_limit: Limit number of samples

    Returns:
        inspect_ai Task
    """
    ds = hf_dataset(
        "Idavidrein/gpqa",
        name=subset,
        split="train",  # GPQA only has train split
        sample_fields=gpqa_record_to_sample,
        limit=sample_limit,
    )

    solvers = []

    if system_prompt:
        solvers.append(system_message(system_prompt))

    solvers.append(multiple_choice())
    solvers.append(generate())

    return Task(
        dataset=ds,
        solver=solvers,
        scorer=mcq_scorer(),
        config={
            "subset": subset,
        },
    )


@task
def gpqa_diamond(sample_limit: int | None = None) -> Task:
    """GPQA Diamond subset evaluation."""
    return science_reasoning_task(subset="gpqa_diamond", sample_limit=sample_limit)
