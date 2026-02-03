"""Tests for math grading utilities."""

from __future__ import annotations

import pytest

from src.evaluation.math_grader import (
    extract_boxed_answer,
    extract_math_answer,
    grade_math_answer,
    is_digit,
    math_equal,
    normalize,
)


class TestIsDigit:
    """Tests for is_digit function."""

    def test_integer(self) -> None:
        """Test integer recognition."""
        is_num, value = is_digit("42")
        assert is_num is True
        assert value == 42.0

    def test_float(self) -> None:
        """Test float recognition."""
        is_num, value = is_digit("3.14")
        assert is_num is True
        assert abs(value - 3.14) < 0.001

    def test_with_commas(self) -> None:
        """Test number with comma separators."""
        is_num, value = is_digit("1,234,567")
        assert is_num is True
        assert value == 1234567.0

    def test_not_a_number(self) -> None:
        """Test non-numeric string."""
        is_num, value = is_digit("hello")
        assert is_num is False
        assert value is None


class TestNormalize:
    """Tests for normalize function."""

    def test_dollar_amount(self) -> None:
        """Test normalizing dollar amounts."""
        result = normalize("$42.50")
        assert result == "42.50"

    def test_percentage(self) -> None:
        """Test normalizing percentages."""
        result = normalize("75%")
        assert result == "75"

    def test_escaped_percentage(self) -> None:
        """Test normalizing LaTeX-escaped percentage."""
        result = normalize("75\\%")
        assert result == "75"


class TestExtractBoxedAnswer:
    """Tests for extract_boxed_answer function."""

    def test_simple_boxed(self) -> None:
        """Test extracting simple boxed answer."""
        text = "The answer is \\boxed{42}"
        result = extract_boxed_answer(text)
        assert result == "42"

    def test_nested_braces(self) -> None:
        """Test extracting boxed answer with nested braces."""
        text = "The answer is \\boxed{\\frac{1}{2}}"
        result = extract_boxed_answer(text)
        assert result == "\\frac{1}{2}"

    def test_no_boxed(self) -> None:
        """Test when no boxed answer present."""
        text = "The answer is 42"
        result = extract_boxed_answer(text)
        assert result is None

    def test_fbox(self) -> None:
        """Test that \\fbox is not supported (only \\boxed is)."""
        text = "The answer is \\fbox{42}"
        result = extract_boxed_answer(text)
        # Note: The implementation only supports \boxed, not \fbox
        assert result is None


class TestExtractMathAnswer:
    """Tests for extract_math_answer function."""

    def test_gsm8k_format(self) -> None:
        """Test GSM8K #### format."""
        text = "Let me calculate...\n#### 42"
        result = extract_math_answer(text)
        assert result == "42"

    def test_natural_language(self) -> None:
        """Test 'the answer is' pattern."""
        text = "Therefore, the answer is 42"
        result = extract_math_answer(text)
        assert result == "42"

    def test_boxed_priority(self) -> None:
        """Test that boxed answer takes priority."""
        text = "The answer is 100 but actually \\boxed{42}"
        result = extract_math_answer(text)
        assert result == "42"


class TestMathEqual:
    """Tests for math_equal function."""

    def test_exact_match(self) -> None:
        """Test exact string match."""
        assert math_equal("42", "42") is True

    def test_numeric_tolerance(self) -> None:
        """Test numeric comparison with tolerance."""
        assert math_equal("3.14159", "3.14159") is True
        assert math_equal("3.1416", "3.14159") is True  # Within tolerance

    def test_percentage_equivalence(self) -> None:
        """Test percentage equivalence."""
        assert math_equal("0.5", "50", include_percentage=True) is True
        assert math_equal("50", "0.5", include_percentage=True) is True

    def test_case_insensitive(self) -> None:
        """Test case-insensitive comparison."""
        assert math_equal("TRUE", "true") is True

    def test_whitespace_handling(self) -> None:
        """Test whitespace is normalized."""
        assert math_equal(" 42 ", "42") is True

    def test_not_equal(self) -> None:
        """Test unequal values."""
        assert math_equal("42", "43") is False
        assert math_equal("hello", "world") is False


class TestGradeMathAnswer:
    """Tests for grade_math_answer function."""

    def test_correct_answer(self) -> None:
        """Test grading a correct answer."""
        response = "Let me solve this step by step.\n#### 42"
        is_correct, feedback = grade_math_answer(response, "42")

        assert is_correct is True
        assert "Correct" in feedback

    def test_incorrect_answer(self) -> None:
        """Test grading an incorrect answer."""
        response = "The answer is 100"
        is_correct, feedback = grade_math_answer(response, "42")

        assert is_correct is False
        assert "Expected" in feedback

    def test_no_answer_found(self) -> None:
        """Test when no answer can be extracted."""
        response = "I cannot solve this problem."
        is_correct, feedback = grade_math_answer(response, "42")

        assert is_correct is False
