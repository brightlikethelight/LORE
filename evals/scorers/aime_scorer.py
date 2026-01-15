"""Scorer for AIME (American Invitational Mathematics Examination) problems.

AIME answers are integers between 0 and 999 (inclusive).
"""

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


def extract_aime_answer(response: str) -> int | None:
    """Extract AIME integer answer (0-999) from response.

    Args:
        response: Model response text

    Returns:
        Integer answer or None if not found/invalid
    """
    # Patterns for AIME-style answers
    patterns = [
        r"(?:answer|result)\s*(?:is|=|:)\s*\*?\*?(\d{1,3})\*?\*?",
        r"\\boxed{(\d{1,3})}",
        r"(?:^|\n)\s*(\d{1,3})\s*$",  # Number alone on line
        r"=\s*(\d{1,3})\s*(?:\.|$)",  # After equals
        r"(?:final answer|therefore)\s*(?:is|:)?\s*(\d{1,3})",
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            try:
                value = int(match.group(1))
                if 0 <= value <= 999:
                    return value
            except ValueError:
                continue

    # Fallback: find last 1-3 digit number in response
    numbers = re.findall(r"\b(\d{1,3})\b", response)
    if numbers:
        try:
            value = int(numbers[-1])
            if 0 <= value <= 999:
                return value
        except ValueError:
            pass

    return None


@scorer(metrics=[accuracy()])
def aime_scorer() -> Scorer:
    """Create a scorer for AIME problems.

    Extracts integer answers (0-999) and compares exactly.

    Returns:
        Scorer instance
    """

    async def score(state: TaskState, target: Target) -> Score:
        response = state.output.completion

        # Extract answer
        extracted = extract_aime_answer(response)

        if extracted is None:
            return Score(
                value="I",
                answer=None,
                explanation="Could not extract valid AIME answer (0-999)",
            )

        # Parse target
        target_str = target.text if hasattr(target, "text") else str(target)
        try:
            target_int = int(target_str.strip())
        except ValueError:
            return Score(
                value="I",
                answer=str(extracted),
                explanation=f"Invalid target format: {target_str}",
            )

        is_correct = extracted == target_int

        return Score(
            value="C" if is_correct else "I",
            answer=str(extracted),
            explanation=f"Predicted: {extracted}, Target: {target_int}",
        )

    return score
