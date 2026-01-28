"""Reward functions for Verl GRPO training.

This module provides reward functions compatible with Verl's trainer:
- MathVerifyReward: Uses math-verify library for symbolic comparison
- SynLogicReward: Rule-based verification for SynLogic tasks
- CodeExecutionReward: Docker-based code execution verification

All reward functions return float values (typically 0.0 or 1.0 for binary rewards).
"""

from __future__ import annotations

import ast
import io
import json
import logging
import os
import re
import signal
import subprocess
import tempfile
from abc import ABC, abstractmethod
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ExecutionTimeoutError(Exception):
    """Raised when code execution times out."""
    pass


def _timeout_handler(signum: int, frame: Any) -> None:
    """Handle execution timeout signal."""
    raise ExecutionTimeoutError("Code execution timed out")


@dataclass
class RewardResult:
    """Result of reward computation.

    Attributes:
        reward: Numeric reward value (0.0 to 1.0)
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


class BaseRewardFunction(ABC):
    """Abstract base class for Verl reward functions."""

    @abstractmethod
    def compute(
        self,
        response: str,
        sample: dict[str, Any],
    ) -> float:
        """Compute reward for a response.

        Args:
            response: Model's generated response
            sample: Dictionary containing 'answer', 'task_type', etc.

        Returns:
            Reward value (0.0 to 1.0)
        """
        pass

    def __call__(
        self,
        response: str,
        sample: dict[str, Any],
    ) -> float:
        """Make the reward function callable."""
        return self.compute(response, sample)


class MathVerifyReward(BaseRewardFunction):
    """Math reward function using math-verify library.

    Uses the math-verify library for robust symbolic comparison of
    mathematical answers, handling various formats like:
    - \\boxed{answer}
    - Decimal/fraction equivalence
    - Expression simplification
    """

    def __init__(
        self,
        tolerance: float = 1e-6,
        timeout: int = 30,
        correct_reward: float = 1.0,
        incorrect_reward: float = 0.0,
    ) -> None:
        """Initialize the math reward function.

        Args:
            tolerance: Numerical tolerance for comparisons
            timeout: Timeout in seconds for verification
            correct_reward: Reward for correct answers
            incorrect_reward: Reward for incorrect answers
        """
        self.tolerance = tolerance
        self.timeout = timeout
        self.correct_reward = correct_reward
        self.incorrect_reward = incorrect_reward

        # Try to import math_verify
        self._math_verify_available = False
        try:
            from math_verify import verify, parse
            self._verify_fn = verify
            self._parse_fn = parse
            self._math_verify_available = True
            logger.info("math-verify library loaded successfully")
        except ImportError:
            logger.warning(
                "math-verify library not available, falling back to regex-based extraction"
            )
            self._verify_fn = None
            self._parse_fn = None

    def extract_boxed_answer(self, text: str) -> str | None:
        """Extract answer from \\boxed{} format.

        Args:
            text: Text containing boxed answer

        Returns:
            Extracted answer or None
        """
        # Match \\boxed{...} with nested braces handling
        patterns = [
            r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",
            r"\\boxed\s*\{([^}]+)\}",
            r"\$\\boxed\{([^}]+)\}\$",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()

        # Also check for final answer patterns
        answer_patterns = [
            r"(?:final\s+)?answer\s*(?:is|:)\s*([^\n.]+)",
            r"(?:therefore|thus|hence|so)\s*,?\s*(?:the\s+)?(?:answer\s+is\s+)?([^\n.]+)",
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    def _verify_with_math_verify(
        self,
        response: str,
        expected: str,
    ) -> bool:
        """Verify using math-verify library.

        Args:
            response: Model response
            expected: Expected answer

        Returns:
            True if answers match
        """
        if not self._math_verify_available:
            return self._verify_with_regex(response, expected)

        try:
            # Extract answer from response
            extracted = self.extract_boxed_answer(response)
            if extracted is None:
                # Try to find any mathematical expression in the response
                extracted = response.strip().split("\n")[-1].strip()

            # Use math-verify: parse both strings then verify
            parsed_extracted = self._parse_fn(extracted)
            parsed_expected = self._parse_fn(expected)
            result = self._verify_fn(parsed_extracted, parsed_expected)
            return result
        except Exception as e:
            logger.debug(f"math-verify failed: {e}, falling back to regex")
            return self._verify_with_regex(response, expected)

    def _verify_with_regex(
        self,
        response: str,
        expected: str,
    ) -> bool:
        """Fallback verification using regex and simple comparison.

        Args:
            response: Model response
            expected: Expected answer

        Returns:
            True if answers match
        """
        extracted = self.extract_boxed_answer(response)
        if extracted is None:
            # Try last line
            extracted = response.strip().split("\n")[-1].strip()

        # Clean both for comparison
        def clean_answer(ans: str) -> str:
            ans = ans.strip()
            # Remove LaTeX formatting
            ans = re.sub(r"\\[a-zA-Z]+", "", ans)
            ans = re.sub(r"[{}$\\]", "", ans)
            ans = re.sub(r"\s+", "", ans)
            return ans.lower()

        extracted_clean = clean_answer(extracted)
        expected_clean = clean_answer(expected)

        # Exact match
        if extracted_clean == expected_clean:
            return True

        # Try numeric comparison
        try:
            extracted_num = float(eval(extracted_clean.replace(",", "")))
            expected_num = float(eval(expected_clean.replace(",", "")))
            return abs(extracted_num - expected_num) < self.tolerance
        except:
            pass

        return False

    def compute(
        self,
        response: str,
        sample: dict[str, Any],
    ) -> float:
        """Compute math reward.

        Args:
            response: Model's generated response
            sample: Dictionary with 'answer' key

        Returns:
            Reward value
        """
        expected = sample.get("answer", "")
        if not expected:
            return self.incorrect_reward

        is_correct = self._verify_with_math_verify(response, expected)
        return self.correct_reward if is_correct else self.incorrect_reward


class SynLogicReward(BaseRewardFunction):
    """Reward function for SynLogic logic reasoning tasks.

    Verifies responses using the expected format:
    - <think>...</think> tags for reasoning
    - <answer>...</answer> tags for final answer

    Reward = 1 if format is correct AND answer matches, else 0.
    """

    def __init__(
        self,
        require_think_tags: bool = True,
        correct_reward: float = 1.0,
        incorrect_reward: float = 0.0,
        format_penalty: float = 0.0,
    ) -> None:
        """Initialize the SynLogic reward function.

        Args:
            require_think_tags: Whether to require <think> tags
            correct_reward: Reward for correct answers
            incorrect_reward: Reward for incorrect answers
            format_penalty: Penalty for missing format (subtracted from correct_reward)
        """
        self.require_think_tags = require_think_tags
        self.correct_reward = correct_reward
        self.incorrect_reward = incorrect_reward
        self.format_penalty = format_penalty

    def extract_answer_tag(self, response: str) -> str | None:
        """Extract content from <answer>...</answer> tags.

        Args:
            response: Model response

        Returns:
            Extracted answer or None
        """
        # Match <answer>...</answer> with various formats
        patterns = [
            r"<answer>\s*(.*?)\s*</answer>",
            r"<answer>(.*?)</answer>",
            r"\[answer\](.*?)\[/answer\]",
            r"Answer:\s*([^\n]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    def has_think_tags(self, response: str) -> bool:
        """Check if response has <think>...</think> tags.

        Args:
            response: Model response

        Returns:
            True if think tags are present
        """
        patterns = [
            r"<think>.*?</think>",
            r"\[think\].*?\[/think\]",
        ]

        for pattern in patterns:
            if re.search(pattern, response, re.DOTALL | re.IGNORECASE):
                return True
        return False

    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison.

        Args:
            answer: Answer string

        Returns:
            Normalized answer
        """
        if not answer:
            return ""
        # Lowercase and strip whitespace
        answer = answer.lower().strip()
        # Remove common punctuation
        answer = re.sub(r"[.,;:!?\"']", "", answer)
        # Normalize whitespace
        answer = re.sub(r"\s+", " ", answer)
        return answer

    def compute(
        self,
        response: str,
        sample: dict[str, Any],
    ) -> float:
        """Compute SynLogic reward.

        Args:
            response: Model's generated response
            sample: Dictionary with 'answer' key and optional 'extra_info'

        Returns:
            Reward value
        """
        expected = sample.get("answer", "")
        if not expected:
            return self.incorrect_reward

        # Check format
        has_format = True
        if self.require_think_tags:
            has_format = self.has_think_tags(response)

        # Extract answer
        extracted = self.extract_answer_tag(response)
        if extracted is None:
            return self.incorrect_reward

        # Normalize and compare
        extracted_normalized = self.normalize_answer(extracted)
        expected_normalized = self.normalize_answer(expected)

        if extracted_normalized == expected_normalized:
            reward = self.correct_reward
            if not has_format and self.format_penalty > 0:
                reward -= self.format_penalty
            return max(reward, self.incorrect_reward)

        # Also check for substring match for multi-word answers
        if expected_normalized in extracted_normalized:
            reward = self.correct_reward
            if not has_format and self.format_penalty > 0:
                reward -= self.format_penalty
            return max(reward, self.incorrect_reward)

        return self.incorrect_reward


class CodeExecutionReward(BaseRewardFunction):
    """Reward function for code execution tasks.

    Executes generated code against test cases using either:
    - Local Docker container (preferred)
    - In-process sandboxed execution (fallback)
    """

    def __init__(
        self,
        use_docker: bool = True,
        docker_image: str = "python:3.10-slim",
        timeout: int = 10,
        memory_limit: str = "256m",
        correct_reward: float = 1.0,
        incorrect_reward: float = 0.0,
        partial_credit: bool = False,
    ) -> None:
        """Initialize the code execution reward function.

        Args:
            use_docker: Whether to use Docker for execution
            docker_image: Docker image to use
            timeout: Execution timeout in seconds
            memory_limit: Docker memory limit
            correct_reward: Reward for passing all tests
            incorrect_reward: Reward for failing tests
            partial_credit: Whether to give partial credit for partial test passes
        """
        self.use_docker = use_docker
        self.docker_image = docker_image
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.correct_reward = correct_reward
        self.incorrect_reward = incorrect_reward
        self.partial_credit = partial_credit

        # Check if Docker is available
        self._docker_available = False
        if use_docker:
            try:
                result = subprocess.run(
                    ["docker", "--version"],
                    capture_output=True,
                    timeout=5,
                )
                self._docker_available = result.returncode == 0
                if self._docker_available:
                    logger.info("Docker available for code execution")
            except (subprocess.SubprocessError, FileNotFoundError):
                logger.warning("Docker not available, falling back to in-process execution")

    def extract_code(self, response: str) -> str | None:
        """Extract Python code from response.

        Args:
            response: Model response

        Returns:
            Extracted code or None
        """
        # Match ```python ... ``` blocks
        patterns = [
            r"```python\s*\n(.*?)```",
            r"```\s*\n(.*?)```",
            r"```py\s*\n(.*?)```",
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                return match.group(1).strip()

        # If no code block, try to find code-like content
        lines = response.split("\n")
        code_lines = []
        in_code = False

        for line in lines:
            # Heuristics for code detection
            if (
                line.strip().startswith("def ")
                or line.strip().startswith("class ")
                or line.strip().startswith("import ")
                or line.strip().startswith("from ")
            ):
                in_code = True

            if in_code:
                code_lines.append(line)

        if code_lines:
            return "\n".join(code_lines)

        return None

    def _execute_with_docker(
        self,
        code: str,
        test_input: str,
    ) -> tuple[str, bool]:
        """Execute code in Docker container.

        Args:
            code: Python code to execute
            test_input: Input to feed to the program

        Returns:
            Tuple of (output, success)
        """
        try:
            # Create temp file with code
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as f:
                f.write(code)
                code_path = f.name

            try:
                # Run in Docker
                cmd = [
                    "docker", "run",
                    "--rm",
                    f"--memory={self.memory_limit}",
                    "--cpus=1",
                    "--network=none",
                    "-i",
                    f"-v{code_path}:/app/code.py:ro",
                    self.docker_image,
                    "python", "/app/code.py",
                ]

                result = subprocess.run(
                    cmd,
                    input=test_input,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )

                return result.stdout, result.returncode == 0
            finally:
                os.unlink(code_path)

        except subprocess.TimeoutExpired:
            return "", False
        except Exception as e:
            logger.debug(f"Docker execution failed: {e}")
            return "", False

    def _execute_in_process(
        self,
        code: str,
        test_input: str,
    ) -> tuple[str, bool]:
        """Execute code in sandboxed process.

        Args:
            code: Python code to execute
            test_input: Input to feed to the program

        Returns:
            Tuple of (output, success)
        """
        # Validate syntax
        try:
            ast.parse(code)
        except SyntaxError:
            return "", False

        # Import common modules for sandbox
        import math
        import itertools
        import functools
        import collections
        import string
        import re as re_module
        import heapq
        import bisect

        # Allowed modules for import statements
        allowed_modules = {
            "math": math,
            "itertools": itertools,
            "functools": functools,
            "collections": collections,
            "string": string,
            "re": re_module,
            "heapq": heapq,
            "bisect": bisect,
        }

        def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            """Safe import that only allows whitelisted modules."""
            if name in allowed_modules:
                return allowed_modules[name]
            raise ImportError(f"Import of '{name}' is not allowed")

        # Set up restricted globals
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
            "input": lambda: test_input.split("\n").pop(0) if test_input else "",
            "__import__": safe_import,
        }

        restricted_globals = {
            "__builtins__": safe_builtins,
            "__name__": "__main__",
            **allowed_modules,  # Pre-load all allowed modules
        }

        # Mock stdin
        original_stdin = os.sys.stdin
        os.sys.stdin = io.StringIO(test_input)

        # Capture output
        stdout_capture = io.StringIO()

        # Set timeout
        old_handler = None
        try:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(self.timeout)
        except (AttributeError, ValueError):
            pass

        try:
            with redirect_stdout(stdout_capture):
                exec(compile(code, "<string>", "exec"), restricted_globals)

            return stdout_capture.getvalue(), True

        except ExecutionTimeoutError:
            return "", False
        except Exception:
            return "", False
        finally:
            try:
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)
            except (AttributeError, ValueError):
                pass
            os.sys.stdin = original_stdin

    def execute_code(
        self,
        code: str,
        test_input: str,
    ) -> tuple[str, bool]:
        """Execute code with appropriate method.

        Args:
            code: Python code to execute
            test_input: Input to feed to the program

        Returns:
            Tuple of (output, success)
        """
        if self.use_docker and self._docker_available:
            return self._execute_with_docker(code, test_input)
        return self._execute_in_process(code, test_input)

    def compute(
        self,
        response: str,
        sample: dict[str, Any],
    ) -> float:
        """Compute code execution reward.

        Args:
            response: Model's generated response
            sample: Dictionary with 'extra_info' containing 'test_cases'

        Returns:
            Reward value
        """
        # Extract code
        code = self.extract_code(response)
        if code is None:
            return self.incorrect_reward

        # Get test cases from extra_info
        extra_info = sample.get("extra_info", {})
        if isinstance(extra_info, str):
            try:
                extra_info = json.loads(extra_info)
            except json.JSONDecodeError:
                extra_info = {}

        test_cases = extra_info.get("test_cases", [])
        if not test_cases:
            # No test cases, just check if code runs
            _, success = self.execute_code(code, "")
            return self.correct_reward if success else self.incorrect_reward

        # Run against test cases
        passed = 0
        total = len(test_cases)

        for test_case in test_cases:
            test_input = test_case.get("input", "")
            expected_output = test_case.get("output", "")

            actual_output, success = self.execute_code(code, test_input)

            if success and actual_output.strip() == expected_output.strip():
                passed += 1

        if passed == total:
            return self.correct_reward

        if self.partial_credit and total > 0:
            return self.incorrect_reward + (self.correct_reward - self.incorrect_reward) * (passed / total)

        return self.incorrect_reward


class MultiTaskReward(BaseRewardFunction):
    """Unified reward function that dispatches to task-specific rewards."""

    def __init__(
        self,
        math_reward: MathVerifyReward | None = None,
        logic_reward: SynLogicReward | None = None,
        code_reward: CodeExecutionReward | None = None,
        default_correct: float = 1.0,
        default_incorrect: float = 0.0,
    ) -> None:
        """Initialize the multi-task reward function.

        Args:
            math_reward: Math reward function
            logic_reward: Logic reward function
            code_reward: Code execution reward function
            default_correct: Default correct reward
            default_incorrect: Default incorrect reward
        """
        self.math_reward = math_reward or MathVerifyReward()
        self.logic_reward = logic_reward or SynLogicReward()
        self.code_reward = code_reward or CodeExecutionReward()
        self.default_correct = default_correct
        self.default_incorrect = default_incorrect

    def compute(
        self,
        response: str,
        sample: dict[str, Any],
    ) -> float:
        """Compute reward based on task type.

        Args:
            response: Model's generated response
            sample: Dictionary with 'task_type' and task-specific fields

        Returns:
            Reward value
        """
        task_type = sample.get("task_type", "math")

        if task_type == "math":
            return self.math_reward.compute(response, sample)
        elif task_type == "logic":
            return self.logic_reward.compute(response, sample)
        elif task_type == "code":
            return self.code_reward.compute(response, sample)
        else:
            # Unknown task type, check if answer appears in response
            expected = sample.get("answer", "")
            if expected and expected.lower() in response.lower():
                return self.default_correct
            return self.default_incorrect


def create_verl_reward_function(
    task_types: list[str] | None = None,
    math_tolerance: float = 1e-6,
    use_docker: bool = True,
    partial_credit: bool = False,
) -> Callable[[list[str], list[dict[str, Any]]], list[float]]:
    """Create a reward function compatible with Verl's trainer interface.

    Verl expects a function that takes lists of responses and samples,
    returning a list of reward values.

    Args:
        task_types: Task types to support (None for all)
        math_tolerance: Tolerance for math comparisons
        use_docker: Whether to use Docker for code execution
        partial_credit: Whether to give partial credit

    Returns:
        Reward function for Verl
    """
    multi_reward = MultiTaskReward(
        math_reward=MathVerifyReward(tolerance=math_tolerance),
        logic_reward=SynLogicReward(),
        code_reward=CodeExecutionReward(use_docker=use_docker, partial_credit=partial_credit),
    )

    def reward_fn(
        responses: list[str],
        samples: list[dict[str, Any]],
    ) -> list[float]:
        """Compute rewards for a batch of responses.

        Args:
            responses: List of model responses
            samples: List of sample dictionaries

        Returns:
            List of reward values
        """
        rewards = []
        for response, sample in zip(responses, samples):
            reward = multi_reward.compute(response, sample)
            rewards.append(reward)
        return rewards

    return reward_fn
