"""Scorer for mathematical answer extraction and verification.

This module uses the comprehensive math grading utilities from
src/evaluation/math_grader.py to avoid code duplication.
"""

from typing import Any

from inspect_ai.scorer import (
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
)
from inspect_ai.solver import TaskState

# Import grading functions from the canonical implementation
from src.evaluation.math_grader import (
    normalize as normalize_answer,
    extract_math_answer as extract_answer,
    math_equal as answers_equivalent,
)


@scorer(metrics=[accuracy()])
def math_answer_scorer() -> Scorer:
    """Create a scorer for mathematical answers.

    Extracts answers from responses and compares them to targets,
    handling various formats and normalizations.

    Returns:
        Scorer instance
    """

    async def score(state: TaskState, target: Target) -> Score:
        # Get model response
        response = state.output.completion

        # Extract answer from response
        extracted = extract_answer(response)

        if extracted is None:
            return Score(
                value="I",  # Incorrect - couldn't extract
                answer=None,
                explanation="Could not extract answer from response",
            )

        # Compare with target
        target_str = target.text if hasattr(target, "text") else str(target)
        is_correct = answers_equivalent(extracted, target_str)

        return Score(
            value="C" if is_correct else "I",
            answer=extracted,
            explanation=f"Extracted: {extracted}, Target: {target_str}",
        )

    return score
