"""Scorer for mathematical answer extraction and verification."""

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


def normalize_answer(answer: str) -> str:
    """Normalize a mathematical answer for comparison.

    Handles:
    - Whitespace stripping
    - Comma removal from numbers
    - Fraction normalization
    - LaTeX cleanup
    """
    if not answer:
        return ""

    # Strip whitespace
    answer = answer.strip()

    # Remove commas from numbers
    answer = re.sub(r"(\d),(\d)", r"\1\2", answer)

    # Remove LaTeX formatting
    answer = re.sub(r"\\text{([^}]*)}", r"\1", answer)
    answer = re.sub(r"\\mathrm{([^}]*)}", r"\1", answer)
    answer = re.sub(r"\\frac{([^}]*)}{([^}]*)}", r"(\1)/(\2)", answer)
    answer = re.sub(r"\$", "", answer)

    # Normalize whitespace
    answer = " ".join(answer.split())

    return answer


def extract_answer(response: str) -> str | None:
    """Extract the final answer from a model response.

    Tries multiple patterns:
    1. #### answer (GSM8K format)
    2. \\boxed{answer} (LaTeX format)
    3. "answer is X" pattern
    4. "= X" at end of response
    """
    patterns = [
        r"####\s*(.*?)(?:\n|$)",  # GSM8K format
        r"\\boxed{([^}]*)}",  # LaTeX boxed
        r"(?:answer|result|solution)\s*(?:is|=|:)\s*([^\n.,]+)",  # Natural language
        r"=\s*([^\n]+)$",  # Equals at end
        r"(\d+(?:\.\d+)?)\s*$",  # Number at end
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()

    return None


def answers_equivalent(pred: str, target: str) -> bool:
    """Check if two answers are mathematically equivalent.

    Args:
        pred: Predicted answer
        target: Target answer

    Returns:
        True if answers are equivalent
    """
    pred_norm = normalize_answer(pred)
    target_norm = normalize_answer(target)

    # Direct string match
    if pred_norm == target_norm:
        return True

    # Try numeric comparison
    try:
        pred_float = float(pred_norm.replace(" ", ""))
        target_float = float(target_norm.replace(" ", ""))
        return abs(pred_float - target_float) < 1e-6
    except ValueError:
        pass

    # Try fraction comparison
    try:
        from fractions import Fraction

        pred_frac = Fraction(pred_norm.replace(" ", ""))
        target_frac = Fraction(target_norm.replace(" ", ""))
        return pred_frac == target_frac
    except (ValueError, ZeroDivisionError):
        pass

    return False


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
