"""Reward functions for GRPO training.

This module provides task-based reward functions for the UltraInteract dataset.
Rewards are computed based on answer correctness using:
- Mathematical equivalence checking for Math_CoT tasks
- Code execution + numeric comparison for Math_PoT tasks
- Code execution for Coding tasks (requires test cases)
- Exact answer matching for Logic tasks
"""

from __future__ import annotations

import ast
import io
import logging
import re
import signal
from abc import ABC, abstractmethod
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from typing import Any, Callable

from src.data.ultrainteract import UITaskType, UltraInteractSample
from src.evaluation.math_grader import (
    extract_math_answer,
    grade_math_answer,
    math_equal,
)

logger = logging.getLogger(__name__)


class ExecutionTimeoutError(Exception):
    """Raised when code execution times out."""
    pass


def _timeout_handler(signum: int, frame: Any) -> None:
    """Handle execution timeout signal."""
    raise ExecutionTimeoutError("Code execution timed out")


def execute_python_code(code: str, timeout: int = 5) -> dict[str, Any]:
    """Execute Python code safely and capture output.

    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds

    Returns:
        Dictionary with 'stdout', 'stderr', 'error', 'success' fields
    """
    result = {
        "stdout": "",
        "stderr": "",
        "error": None,
        "success": False,
    }

    # Validate code syntax
    try:
        ast.parse(code)
    except SyntaxError as e:
        result["error"] = f"SyntaxError: {e}"
        return result

    # Set up safe globals (restricted builtins)
    safe_builtins = {
        "abs": abs, "all": all, "any": any, "bin": bin, "bool": bool,
        "chr": chr, "dict": dict, "divmod": divmod, "enumerate": enumerate,
        "filter": filter, "float": float, "format": format, "frozenset": frozenset,
        "hex": hex, "int": int, "isinstance": isinstance, "iter": iter,
        "len": len, "list": list, "map": map, "max": max, "min": min,
        "oct": oct, "ord": ord, "pow": pow, "print": print, "range": range,
        "repr": repr, "reversed": reversed, "round": round, "set": set,
        "slice": slice, "sorted": sorted, "str": str, "sum": sum,
        "tuple": tuple, "type": type, "zip": zip,
        "__import__": __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__,
    }

    restricted_globals = {
        "__builtins__": safe_builtins,
        "__name__": "__main__",
    }

    # Import commonly needed modules
    try:
        import math
        import itertools
        import functools
        import collections
        restricted_globals["math"] = math
        restricted_globals["itertools"] = itertools
        restricted_globals["functools"] = functools
        restricted_globals["collections"] = collections
    except ImportError:
        pass

    # Capture output
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    # Set timeout (Unix only)
    old_handler = None
    try:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)
    except (AttributeError, ValueError):
        # Windows or signal not available
        pass

    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(compile(code, "<string>", "exec"), restricted_globals)

        result["stdout"] = stdout_capture.getvalue()
        result["stderr"] = stderr_capture.getvalue()
        result["success"] = True

    except ExecutionTimeoutError:
        result["error"] = "Execution timed out"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}"
    finally:
        # Reset alarm
        try:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
        except (AttributeError, ValueError):
            pass

    return result


def extract_code_output(execution_result: dict[str, Any]) -> str | None:
    """Extract the final output value from executed code.

    Args:
        execution_result: Result from execute_python_code

    Returns:
        Extracted output string or None if extraction failed
    """
    if not execution_result["success"]:
        return None

    stdout = execution_result["stdout"].strip()
    if stdout:
        # Get the last printed line (typically the answer)
        lines = stdout.strip().split("\n")
        return lines[-1].strip()

    return None


@dataclass
class RewardResult:
    """Result of reward computation.

    Attributes:
        reward: Numeric reward value
        is_correct: Whether the answer is correct
        feedback: Human-readable feedback
        extracted_answer: Extracted answer from response
        expected_answer: Expected answer
    """

    reward: float
    is_correct: bool
    feedback: str
    extracted_answer: str | None = None
    expected_answer: str | None = None


class RewardFunction(ABC):
    """Abstract base class for reward functions."""

    @abstractmethod
    def __call__(
        self,
        response: str,
        sample: UltraInteractSample | dict[str, Any],
    ) -> RewardResult:
        """Compute reward for a response.

        Args:
            response: Model's generated response
            sample: UltraInteract sample or dict with ground truth

        Returns:
            RewardResult with reward value and metadata
        """
        pass


class TaskRewardFunction(RewardFunction):
    """Task-aware reward function for UltraInteract.

    Dispatches to appropriate grading logic based on task type:
    - Math_CoT/Math_PoT: Symbolic mathematical equivalence
    - Coding: Code structure and execution (when available)
    - Logic: Answer extraction and semantic matching
    """

    def __init__(
        self,
        correct_reward: float = 1.0,
        incorrect_reward: float = 0.0,
        math_tolerance: float = 1e-4,
        partial_credit: bool = False,
    ) -> None:
        """Initialize the reward function.

        Args:
            correct_reward: Reward for correct answers
            incorrect_reward: Reward for incorrect answers
            math_tolerance: Tolerance for mathematical comparisons
            partial_credit: Whether to award partial credit
        """
        self.correct_reward = correct_reward
        self.incorrect_reward = incorrect_reward
        self.math_tolerance = math_tolerance
        self.partial_credit = partial_credit

    def __call__(
        self,
        response: str,
        sample: UltraInteractSample | dict[str, Any],
    ) -> RewardResult:
        """Compute reward based on task type.

        Args:
            response: Model's generated response
            sample: UltraInteract sample or dict with ground truth

        Returns:
            RewardResult with reward value and metadata
        """
        # Extract task type and expected answer
        if isinstance(sample, UltraInteractSample):
            task_type = sample.task_type
            expected = sample.final_answer or sample._extract_answer_from_response()
        else:
            task_type_str = sample.get("task_type") or sample.get(
                "additional_context", {}
            ).get("task_type", "Coding")
            task_type = (
                UITaskType.from_string(task_type_str)
                if isinstance(task_type_str, str)
                else task_type_str
            )
            expected = sample.get("answer") or sample.get("final_answer", "")

        # Dispatch to task-specific grader
        if task_type in (UITaskType.MATH_COT, UITaskType.MATH_POT):
            return self._grade_math(response, expected, task_type)
        elif task_type == UITaskType.CODING:
            return self._grade_coding(response, expected)
        elif task_type == UITaskType.LOGIC:
            return self._grade_logic(response, expected)
        else:
            # Default: string matching
            return self._grade_default(response, expected)

    def _grade_math(
        self,
        response: str,
        expected: str,
        task_type: UITaskType,
    ) -> RewardResult:
        """Grade math answers using symbolic equivalence or code execution.

        For Math_CoT: Extract answer and compare symbolically.
        For Math_PoT: Execute the Python code and compare output to expected.

        Args:
            response: Model's response
            expected: Expected answer
            task_type: Math_CoT or Math_PoT

        Returns:
            RewardResult
        """
        if not expected:
            return RewardResult(
                reward=self.incorrect_reward,
                is_correct=False,
                feedback="No expected answer provided",
            )

        # For Math_PoT, try to execute the code and compare output
        if task_type == UITaskType.MATH_POT:
            return self._grade_math_pot_execution(response, expected)

        # For Math_CoT, use symbolic comparison
        is_correct, feedback = grade_math_answer(
            response, expected, tolerance=self.math_tolerance
        )

        # Extract the answer that was compared
        extracted = extract_math_answer(response)

        return RewardResult(
            reward=self.correct_reward if is_correct else self.incorrect_reward,
            is_correct=is_correct,
            feedback=feedback,
            extracted_answer=extracted,
            expected_answer=expected,
        )

    def _grade_math_pot_execution(
        self,
        response: str,
        expected: str,
    ) -> RewardResult:
        """Grade Math_PoT by executing code and comparing output.

        Args:
            response: Model's response containing Python code
            expected: Expected numeric answer

        Returns:
            RewardResult
        """
        # Extract code block from response
        code_match = re.search(r"```(?:python)?\s*\n?(.*?)```", response, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
        else:
            # Try to find code-like content without markdown
            code = response.strip()

        if not code:
            return RewardResult(
                reward=self.incorrect_reward,
                is_correct=False,
                feedback="No code found in response",
                expected_answer=expected,
            )

        # Execute the code
        exec_result = execute_python_code(code, timeout=5)

        if not exec_result["success"]:
            return RewardResult(
                reward=self.incorrect_reward,
                is_correct=False,
                feedback=f"Code execution failed: {exec_result['error']}",
                extracted_answer=None,
                expected_answer=expected,
            )

        # Extract output
        output = extract_code_output(exec_result)

        if output is None:
            return RewardResult(
                reward=self.incorrect_reward,
                is_correct=False,
                feedback="No output produced by code",
                extracted_answer=None,
                expected_answer=expected,
            )

        # Compare output to expected using math_equal
        is_correct = math_equal(output, expected, tolerance=self.math_tolerance)

        return RewardResult(
            reward=self.correct_reward if is_correct else self.incorrect_reward,
            is_correct=is_correct,
            feedback="Correct" if is_correct else f"Expected {expected}, got {output}",
            extracted_answer=output,
            expected_answer=expected,
        )

    def _grade_coding(
        self,
        response: str,
        expected: str,
        test_cases: list[tuple[str, str]] | None = None,
    ) -> RewardResult:
        """Grade coding answers using execution against test cases.

        If test cases are provided, executes code against them.
        Otherwise, checks if code runs without errors (syntax/runtime).

        Args:
            response: Model's response
            expected: Expected code/answer (reference solution)
            test_cases: Optional list of (input, expected_output) tuples

        Returns:
            RewardResult
        """
        # Extract code block from response
        code_match = re.search(r"```(?:python)?\s*\n?(.*?)```", response, re.DOTALL)
        response_code = code_match.group(1).strip() if code_match else response.strip()

        if not response_code:
            return RewardResult(
                reward=self.incorrect_reward,
                is_correct=False,
                feedback="No code found in response",
                expected_answer=expected[:200] if expected else None,
            )

        # Check for basic code structure
        has_def = bool(re.search(r"\bdef\s+\w+\s*\(", response_code))
        has_class = bool(re.search(r"\bclass\s+\w+", response_code))
        has_code = has_def or has_class or bool(
            re.search(r"\b(for|while|if|return)\b", response_code)
        )

        if not has_code:
            return RewardResult(
                reward=self.incorrect_reward,
                is_correct=False,
                feedback="No valid code structure found",
                extracted_answer=response_code[:200],
                expected_answer=expected[:200] if expected else None,
            )

        # If test cases provided, use execution-based evaluation
        if test_cases:
            return self._grade_coding_with_tests(response_code, test_cases)

        # Otherwise, check if code compiles/runs without errors
        return self._grade_coding_syntax(response_code, expected)

    def _grade_coding_with_tests(
        self,
        code: str,
        test_cases: list[tuple[str, str]],
    ) -> RewardResult:
        """Grade code by executing against test cases.

        Args:
            code: Python code to test
            test_cases: List of (input, expected_output) tuples

        Returns:
            RewardResult
        """
        passed = 0
        total = len(test_cases)
        failed_cases = []

        for test_input, expected_output in test_cases:
            # Create code that feeds input and captures output
            test_code = f'''
import sys
from io import StringIO

# Capture stdin
sys.stdin = StringIO("""{test_input}""")

# User code
{code}
'''
            result = execute_python_code(test_code, timeout=5)

            if result["success"]:
                actual_output = result["stdout"].strip()
                expected_clean = expected_output.strip()

                if actual_output == expected_clean:
                    passed += 1
                else:
                    failed_cases.append((test_input[:50], expected_clean[:50], actual_output[:50]))
            else:
                failed_cases.append((test_input[:50], "N/A", f"Error: {result['error'][:50]}"))

        is_correct = passed == total
        pass_rate = passed / total if total > 0 else 0

        if is_correct:
            feedback = f"All {total} test cases passed"
        else:
            feedback = f"Passed {passed}/{total} tests"
            if failed_cases:
                feedback += f". First failure: input={failed_cases[0][0]}, expected={failed_cases[0][1]}, got={failed_cases[0][2]}"

        return RewardResult(
            reward=self.correct_reward if is_correct else self.incorrect_reward,
            is_correct=is_correct,
            feedback=feedback,
            extracted_answer=f"pass_rate={pass_rate:.2%}",
        )

    def _grade_coding_syntax(
        self,
        code: str,
        expected: str | None,
    ) -> RewardResult:
        """Grade code by checking if it compiles and runs without errors.

        NOTE: This is a fallback when test cases are not available.
        It only checks syntax and basic runtime validity, not correctness.

        Args:
            code: Python code to check
            expected: Reference solution (for logging only)

        Returns:
            RewardResult
        """
        # First check: does the code parse (syntax check)?
        try:
            ast.parse(code)
        except SyntaxError as e:
            return RewardResult(
                reward=self.incorrect_reward,
                is_correct=False,
                feedback=f"Syntax error: {e}",
                extracted_answer=code[:200],
                expected_answer=expected[:200] if expected else None,
            )

        # Second check: does it execute without errors (with empty input)?
        result = execute_python_code(code, timeout=5)

        if result["success"]:
            return RewardResult(
                reward=self.correct_reward,
                is_correct=True,
                feedback="Code executes without errors",
                extracted_answer=code[:200],
                expected_answer=expected[:200] if expected else None,
            )
        else:
            # Distinguish between runtime errors and missing input
            error = result["error"] or ""
            if "EOFError" in error or "input" in error.lower():
                # Code probably needs input - that's okay
                return RewardResult(
                    reward=self.correct_reward,
                    is_correct=True,
                    feedback="Code compiles (needs input)",
                    extracted_answer=code[:200],
                    expected_answer=expected[:200] if expected else None,
                )
            else:
                return RewardResult(
                    reward=self.incorrect_reward,
                    is_correct=False,
                    feedback=f"Runtime error: {error[:100]}",
                    extracted_answer=code[:200],
                    expected_answer=expected[:200] if expected else None,
                )

    def _grade_logic(
        self,
        response: str,
        expected: str,
    ) -> RewardResult:
        """Grade logic answers using exact answer matching.

        Extracts the answer from the response and compares to expected answer
        using exact string matching (with normalization).

        Args:
            response: Model's response
            expected: Expected answer

        Returns:
            RewardResult
        """
        if not expected:
            return RewardResult(
                reward=self.incorrect_reward,
                is_correct=False,
                feedback="No expected answer provided",
            )

        # Normalize expected answer
        expected_normalized = self._normalize_logic_answer(expected)

        # Extract answer from response using common patterns
        extracted_answer = self._extract_logic_answer(response)

        if extracted_answer:
            # Normalize extracted answer
            extracted_normalized = self._normalize_logic_answer(extracted_answer)

            # Exact match after normalization
            if extracted_normalized == expected_normalized:
                return RewardResult(
                    reward=self.correct_reward,
                    is_correct=True,
                    feedback="Correct answer",
                    extracted_answer=extracted_answer,
                    expected_answer=expected,
                )

            # Check if one contains the other (for multi-word answers)
            if expected_normalized in extracted_normalized or extracted_normalized in expected_normalized:
                return RewardResult(
                    reward=self.correct_reward,
                    is_correct=True,
                    feedback="Answer matches (substring)",
                    extracted_answer=extracted_answer,
                    expected_answer=expected,
                )

        # Fallback: check if expected answer appears directly in response
        response_normalized = self._normalize_logic_answer(response)
        if expected_normalized in response_normalized:
            return RewardResult(
                reward=self.correct_reward,
                is_correct=True,
                feedback="Answer found in response",
                extracted_answer=expected,
                expected_answer=expected,
            )

        return RewardResult(
            reward=self.incorrect_reward,
            is_correct=False,
            feedback=f"Expected '{expected[:50]}', got '{extracted_answer or 'no answer extracted'}'",
            extracted_answer=extracted_answer,
            expected_answer=expected,
        )

    def _normalize_logic_answer(self, text: str) -> str:
        """Normalize a logic answer for comparison.

        - Lowercase
        - Remove punctuation and quotes
        - Normalize whitespace
        - Strip

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        text = text.lower()
        text = re.sub(r'["\'\.,;:!?\-\(\)]', "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _extract_logic_answer(self, response: str) -> str | None:
        """Extract the answer from a logic response.

        Args:
            response: Model's response

        Returns:
            Extracted answer or None
        """
        # Try common answer patterns in order of specificity
        answer_patterns = [
            r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]+([^\.\n]+)",
            r"Answer:\s*([^\.\n]+)",
            r"(?:therefore|thus|hence|so)[,\s]+(?:the\s+)?(?:answer\s+is\s+)?([^\.\n]+)",
            r"(?:In\s+)?conclusion[,:\s]+([^\.\n]+)",
            r"(?:we\s+can\s+conclude\s+that\s+)([^\.\n]+)",
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                answer = match.group(1).strip()
                # Remove trailing punctuation
                answer = re.sub(r"[\.!]+$", "", answer).strip()
                if answer:
                    return answer

        return None

    def _grade_default(
        self,
        response: str,
        expected: str,
    ) -> RewardResult:
        """Default grading: simple string matching.

        Args:
            response: Model's response
            expected: Expected answer

        Returns:
            RewardResult
        """
        if not expected:
            return RewardResult(
                reward=self.incorrect_reward,
                is_correct=False,
                feedback="No expected answer provided",
            )

        is_correct = expected.lower().strip() in response.lower()

        return RewardResult(
            reward=self.correct_reward if is_correct else self.incorrect_reward,
            is_correct=is_correct,
            feedback="Correct" if is_correct else "Incorrect",
            expected_answer=expected,
        )


def compute_reward(
    response: str,
    sample: UltraInteractSample | dict[str, Any],
    correct_reward: float = 1.0,
    incorrect_reward: float = 0.0,
    math_tolerance: float = 1e-4,
) -> float:
    """Convenience function to compute reward for a response.

    Args:
        response: Model's generated response
        sample: UltraInteract sample or dict with ground truth
        correct_reward: Reward for correct answers
        incorrect_reward: Reward for incorrect answers
        math_tolerance: Tolerance for mathematical comparisons

    Returns:
        Numeric reward value
    """
    reward_fn = TaskRewardFunction(
        correct_reward=correct_reward,
        incorrect_reward=incorrect_reward,
        math_tolerance=math_tolerance,
    )
    result = reward_fn(response, sample)
    return result.reward


def create_reward_function(
    correct_reward: float = 1.0,
    incorrect_reward: float = 0.0,
    math_tolerance: float = 1e-4,
    partial_credit: bool = False,
) -> Callable[[str, Any], float]:
    """Create a reward function for TRL GRPO trainer.

    Args:
        correct_reward: Reward for correct answers
        incorrect_reward: Reward for incorrect answers
        math_tolerance: Tolerance for mathematical comparisons
        partial_credit: Whether to use partial credit

    Returns:
        Callable that takes (response, sample) and returns float reward
    """
    reward_fn = TaskRewardFunction(
        correct_reward=correct_reward,
        incorrect_reward=incorrect_reward,
        math_tolerance=math_tolerance,
        partial_credit=partial_credit,
    )

    def _reward(response: str, sample: Any) -> float:
        return reward_fn(response, sample).reward

    return _reward
