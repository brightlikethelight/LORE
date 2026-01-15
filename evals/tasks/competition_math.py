"""Competition mathematics evaluation task using AIME problems.

Evaluates advanced mathematical problem-solving on American Invitational
Mathematics Examination (AIME) problems.
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.solver import generate, system_message

from evals.scorers.aime_scorer import aime_scorer


def aime_record_to_sample(record: dict) -> Sample:
    """Convert AIME record to inspect_ai Sample."""
    return Sample(
        input=record["problem"],
        target=str(record["answer"]),  # AIME answers are integers 0-999
        metadata={
            "year": record.get("year"),
            "problem_number": record.get("problem_number"),
            "competition": record.get("competition", "AIME"),
        },
    )


@task
def competition_math_task(
    system_prompt: str | None = None,
    optimized_prompt: str | None = None,
    sample_limit: int | None = None,
) -> Task:
    """Create competition mathematics evaluation task.

    Args:
        system_prompt: Optional system prompt
        optimized_prompt: Optional optimized reasoning prompt
        sample_limit: Limit number of samples

    Returns:
        inspect_ai Task
    """
    ds = hf_dataset(
        "AI-MO/aimo-validation-aime",
        split="train",
        sample_fields=aime_record_to_sample,
        limit=sample_limit,
    )

    solvers = []

    # Default system prompt for competition math
    default_system = """You are an expert mathematician solving competition problems.
Provide your reasoning step by step, then give your final answer as an integer between 0 and 999.
Format your final answer as: The answer is [number]"""

    solvers.append(system_message(system_prompt or default_system))
    solvers.append(generate())

    return Task(
        dataset=ds,
        solver=solvers,
        scorer=aime_scorer(),
        config={
            "optimized_prompt": optimized_prompt,
        },
    )


@task
def aime_eval(sample_limit: int | None = None) -> Task:
    """AIME evaluation task."""
    return competition_math_task(sample_limit=sample_limit)
