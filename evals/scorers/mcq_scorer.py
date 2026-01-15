"""Scorer for multiple choice question evaluation."""

import re
from typing import Any

from inspect_ai.scorer import (
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
)
from inspect_ai.solver import TaskState


def extract_mcq_answer(response: str) -> str | None:
    """Extract multiple choice answer (A, B, C, D) from response.

    Args:
        response: Model response text

    Returns:
        Extracted letter answer or None
    """
    # Pattern priorities (most specific first)
    patterns = [
        r"(?:answer|choice|option)\s*(?:is|:)\s*\(?([A-Da-d])\)?",
        r"\b([A-Da-d])\)\s*is\s*(?:correct|right|the answer)",
        r"^\s*\(?([A-Da-d])\)?[\.\):]?\s*$",  # Just the letter
        r"(?:select|choose|pick)\s*\(?([A-Da-d])\)?",
        r"\*\*([A-Da-d])\*\*",  # Bold letter
        r"(?:^|\n)\s*([A-Da-d])[\.\)]",  # Letter at start of line
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()

    # Fallback: find any standalone letter
    letters = re.findall(r"\b([A-Da-d])\b", response)
    if letters:
        # Return most common or last occurrence
        return letters[-1].upper()

    return None


@scorer(metrics=[accuracy()])
def mcq_scorer() -> Scorer:
    """Create a scorer for multiple choice questions.

    Extracts letter answers (A-D) and compares to target.

    Returns:
        Scorer instance
    """

    async def score(state: TaskState, target: Target) -> Score:
        response = state.output.completion

        # Extract answer
        extracted = extract_mcq_answer(response)

        if extracted is None:
            return Score(
                value="I",
                answer=None,
                explanation="Could not extract MCQ answer from response",
            )

        # Compare with target
        target_str = target.text if hasattr(target, "text") else str(target)
        target_letter = target_str.strip().upper()

        is_correct = extracted == target_letter

        return Score(
            value="C" if is_correct else "I",
            answer=extracted,
            explanation=f"Selected: {extracted}, Correct: {target_letter}",
        )

    return score
