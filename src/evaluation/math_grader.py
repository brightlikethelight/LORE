"""Mathematical answer grading using sympy-based equivalence checking.

This module provides robust mathematical equivalence checking adapted from
the Eurus evaluation code (Eurus/eval/utils/grader.py).

Key features:
- Numerical comparison with configurable tolerance
- Symbolic equivalence via sympy.simplify
- LaTeX expression parsing
- Handles special cases: percentages, pi, matrices, intervals
"""

from __future__ import annotations

import contextlib
import math
import re
import signal
from math import isclose
from typing import Union

# Try to import sympy, fall back to basic comparison if not available
try:
    import sympy
    from sympy import N, simplify
    from sympy.parsing.latex import parse_latex
    from sympy.parsing.sympy_parser import parse_expr

    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False


class TimeoutException(Exception):
    """Exception raised when operation times out."""

    pass


@contextlib.contextmanager
def time_limit(seconds: float):
    """Context manager for timeout on Unix systems."""

    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    # Only use signal-based timeout on Unix
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        signal.signal(signal.SIGALRM, signal_handler)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    except (AttributeError, ValueError):
        # signal.setitimer not available (Windows) or invalid value
        yield


def is_digit(s) -> tuple[bool, float | None]:
    """Check if string represents a number."""
    try:
        if "{,}" in str(s):
            num = float(str(s).replace("{,}", ""))
            return True, num
        num = float(str(s).replace(",", ""))
        return True, num
    except (ValueError, TypeError):
        return False, None


def normalize(answer: str, pi: float = math.pi) -> str | float:
    """Normalize answer string for comparison."""
    if not isinstance(answer, str):
        return answer

    # Remove dollar sign prefix from monetary amounts
    if bool(re.match(r"\$\d+(\.\d+)?", answer)):
        return answer[1:]

    # Remove percentage suffix
    if bool(re.match(r"^\d+(\.\d+)?%$", answer)) or bool(
        re.match(r"^\d+(\.\d+)?\\%$", answer)
    ):
        return answer.replace("\\%", "").replace("%", "")

    # Handle base notation (e.g., "10_2" for binary)
    answer = handle_base(answer)

    # Handle pi expressions
    answer = handle_pi(answer, pi)

    return answer


def handle_base(x: str | float) -> str | float:
    """Handle base notation in numbers."""
    if isinstance(x, str) and "_" in x:
        try:
            x = x.split("_")[0]
            return int(float(x))
        except (ValueError, TypeError):
            return x
    return x


def handle_pi(string: str | float, pi: float) -> str | float:
    """Handle pi expressions in answers."""
    if not isinstance(string, str) or "\\pi" not in string:
        return string

    # Replace \pi with numeric value
    idx = string.find("\\pi")
    while idx != -1:
        if idx > 0 and string[idx - 1].isdigit():
            string = string[:idx] + f"*{pi}" + string[idx + 3 :]
        else:
            string = string[:idx] + f"1*{pi}" + string[idx + 3 :]
        idx = string.find("\\pi", idx + 1)

    try:
        return eval(string)
    except Exception:
        return string


def format_intervals(prediction: str) -> str:
    """Format sympy interval notation to standard interval notation."""
    patterns = {
        "Interval(": r"^Interval\((.*)\)$",
        "Interval.Ropen(": r"^Interval\.Ropen\((.*)\)$",
        "Interval.Lopen(": r"^Interval\.Lopen\((.*)\)$",
        "Interval.open(": r"^Interval\.open\((.*)\)$",
    }

    for key, pattern in patterns.items():
        match = re.match(pattern, prediction)
        if match:
            inner_content = match.group(1)
            if key == "Interval(":
                return f"[{inner_content}]"
            elif key == "Interval.Ropen(":
                return f"[{inner_content})"
            elif key == "Interval.Lopen(":
                return f"({inner_content}]"
            elif key == "Interval.open(":
                return f"({inner_content})"

    return prediction


def symbolic_equal(
    a: str, b: str, tolerance: float = 1e-4, timeout: float = 5.0
) -> bool:
    """Check symbolic equality using sympy."""
    if not HAS_SYMPY:
        return False

    def _parse(s):
        for f in [parse_expr, parse_latex]:
            try:
                with time_limit(timeout):
                    return f(s)
            except Exception:
                pass
        return s

    a_parsed = _parse(a)
    b_parsed = _parse(b)

    try:
        with time_limit(timeout):
            if simplify(a_parsed - b_parsed) == 0:
                return True
    except Exception:
        pass

    try:
        with time_limit(timeout):
            if isclose(N(a_parsed), N(b_parsed), rel_tol=tolerance):
                return True
    except Exception:
        pass

    return False


def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    tolerance: float = 1e-4,
    timeout: float = 5.0,
    pi: float = math.pi,
) -> bool:
    """Check if prediction equals reference mathematically.

    This function checks equality in multiple ways:
    1. Direct string comparison
    2. Numerical comparison (with tolerance)
    3. Symbolic comparison using sympy

    Args:
        prediction: Model's predicted answer
        reference: Ground truth answer
        include_percentage: Whether to check percentage equivalence (e.g., 0.5 == 50%)
        tolerance: Relative tolerance for numerical comparison
        timeout: Timeout for symbolic evaluation
        pi: Value of pi to use

    Returns:
        True if answers are mathematically equivalent
    """
    prediction = normalize(prediction, pi)
    reference = normalize(reference, pi)

    # Handle long strings
    if isinstance(prediction, str) and len(prediction) > 1000:
        prediction = prediction[:1000]

    # 0. Direct string comparison
    if isinstance(prediction, str) and isinstance(reference, str):
        if prediction.strip().lower() == reference.strip().lower():
            return True
        if prediction.replace(" ", "") == reference.replace(" ", ""):
            return True

    # 1. Numerical comparison
    try:
        pred_is_digit, pred_num = is_digit(prediction)
        ref_is_digit, ref_num = is_digit(reference)

        if pred_is_digit and ref_is_digit and pred_num is not None and ref_num is not None:
            # Check with percentage variations if enabled
            if include_percentage:
                gt_results = [ref_num / 100, ref_num, ref_num * 100]
            else:
                gt_results = [ref_num]

            for item in gt_results:
                try:
                    if isclose(item, pred_num, rel_tol=tolerance):
                        return True
                except Exception:
                    continue
            return False
    except Exception:
        pass

    # Handle empty/None predictions
    if not prediction and prediction not in [0, False]:
        return False

    # 2. Symbolic comparison
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    # Format interval notation
    prediction = format_intervals(prediction)

    pred_str, ref_str = prediction, reference

    # Handle brackets
    if (
        prediction.startswith("[")
        and prediction.endswith("]")
        and not reference.startswith("(")
    ) or (
        prediction.startswith("(")
        and prediction.endswith(")")
        and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")

    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")

    if pred_str == ref_str:
        return True

    # Handle lists/tuples
    if (
        prediction
        and reference
        and prediction[0] in "(["
        and prediction[-1] in ")]"
        and prediction[0] == reference[0]
        and prediction[-1] == reference[-1]
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(
                math_equal(pred_pt.strip(), ref_pt.strip(), include_percentage, tolerance)
                for pred_pt, ref_pt in zip(pred_parts, ref_parts)
            ):
                return True

    # Handle comma-separated values
    if "," in prediction and "," in reference:
        pred_parts = [item.strip() for item in prediction.split(",")]
        ref_parts = [item.strip() for item in reference.split(",")]
        if len(pred_parts) == len(ref_parts):
            if all(
                math_equal(pred_parts[i], ref_parts[i], include_percentage, tolerance)
                for i in range(len(pred_parts))
            ):
                return True

    return symbolic_equal(prediction, reference, tolerance, timeout)


def extract_boxed_answer(text: str) -> str | None:
    """Extract answer from \\boxed{} expression."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(text):
        if text[i] == "{":
            num_left_braces_open += 1
        if text[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None

    retval = text[idx : right_brace_idx + 1]
    left = "\\boxed{"

    try:
        assert retval[: len(left)] == left
        assert retval[-1] == "}"
        return retval[len(left) : -1]
    except (AssertionError, IndexError):
        return None


def extract_math_answer(response: str) -> str | None:
    """Extract answer from math response using common patterns.

    Args:
        response: Model's full response

    Returns:
        Extracted answer or None
    """
    # Try boxed first
    boxed = extract_boxed_answer(response)
    if boxed:
        return boxed

    # Common answer patterns
    patterns = [
        r"####\s*(.+?)$",
        r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]+([^\n.]+)",
        r"(?:therefore|thus|so)[,\s]+(?:the\s+)?answer\s+is[:\s]+([^\n.]+)",
        r"=\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*$",
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()

    return None


def grade_math_answer(
    response: str,
    reference: str,
    tolerance: float = 1e-4,
) -> tuple[bool, str]:
    """Grade a math answer against reference.

    Args:
        response: Model's full response
        reference: Ground truth answer
        tolerance: Numerical tolerance

    Returns:
        Tuple of (is_correct, feedback)
    """
    # Extract answer from response
    extracted = extract_math_answer(response)

    if extracted is None:
        # Try to find any number in the response
        numbers = re.findall(r"[+-]?\d+(?:,\d{3})*(?:\.\d+)?", response)
        if numbers:
            extracted = numbers[-1]  # Use last number as likely answer
        else:
            return False, "No answer found in response"

    # Check equivalence
    is_correct = math_equal(extracted, reference, tolerance=tolerance)

    if is_correct:
        return True, f"Correct: {extracted}"
    else:
        return False, f"Expected {reference}, got {extracted}"
