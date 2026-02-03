"""Tests for text extraction utilities."""

from __future__ import annotations

import pytest


class TestExtractReasoningAndAnswer:
    """Tests for _extract_reasoning_and_answer function."""

    def test_extract_xml_tags(self) -> None:
        """Test extraction from <thinking> and <answer> tags."""
        from src.models.frontier_api import _extract_reasoning_and_answer

        text = "<thinking>This is my reasoning.</thinking>\n<answer>42</answer>"
        reasoning, answer = _extract_reasoning_and_answer(text)

        assert reasoning == "This is my reasoning."
        assert answer == "42"

    def test_extract_multiline_xml(self) -> None:
        """Test extraction from multiline XML tags."""
        from src.models.frontier_api import _extract_reasoning_and_answer

        text = """<thinking>
Step 1: First step
Step 2: Second step
</thinking>

<answer>The answer is 42</answer>"""
        reasoning, answer = _extract_reasoning_and_answer(text)

        assert "Step 1" in reasoning
        assert "Step 2" in reasoning
        assert answer == "The answer is 42"

    def test_extract_json_format(self) -> None:
        """Test extraction from JSON format."""
        from src.models.frontier_api import _extract_reasoning_and_answer

        text = '{"reasoning": "I calculated carefully", "answer": "100"}'
        reasoning, answer = _extract_reasoning_and_answer(text)

        assert reasoning == "I calculated carefully"
        assert answer == "100"

    def test_extract_text_markers(self) -> None:
        """Test extraction from text markers like 'Reasoning:' and 'Answer:'."""
        from src.models.frontier_api import _extract_reasoning_and_answer

        text = "Reasoning: I thought about it carefully.\nAnswer: 42"
        reasoning, answer = _extract_reasoning_and_answer(text)

        assert "thought about it" in reasoning
        assert answer == "42"

    def test_fallback_to_full_text(self) -> None:
        """Test fallback when no patterns match."""
        from src.models.frontier_api import _extract_reasoning_and_answer

        text = "Just a plain response with no structure."
        reasoning, answer = _extract_reasoning_and_answer(text)

        assert reasoning == ""
        assert answer == "Just a plain response with no structure."


class TestMaybeExtractJson:
    """Tests for JSON extraction utility."""

    def test_extract_plain_json(self) -> None:
        """Test extraction of plain JSON object."""
        from src.models.frontier_api import _maybe_extract_json

        text = '{"key": "value", "number": 42}'
        result = _maybe_extract_json(text)

        assert result is not None
        assert result["key"] == "value"
        assert result["number"] == 42

    def test_extract_json_with_markdown(self) -> None:
        """Test extraction of JSON wrapped in markdown code block."""
        from src.models.frontier_api import _maybe_extract_json

        text = """```json
{"reasoning": "test", "answer": "42"}
```"""
        result = _maybe_extract_json(text)

        assert result is not None
        assert result["reasoning"] == "test"

    def test_returns_none_for_non_json(self) -> None:
        """Test that non-JSON text returns None."""
        from src.models.frontier_api import _maybe_extract_json

        text = "This is not JSON"
        result = _maybe_extract_json(text)

        assert result is None

    def test_returns_none_for_json_array(self) -> None:
        """Test that JSON arrays return None (we expect objects)."""
        from src.models.frontier_api import _maybe_extract_json

        text = "[1, 2, 3]"
        result = _maybe_extract_json(text)

        assert result is None


class TestOpenWeightsExtraction:
    """Tests for open weights model text extraction."""

    def test_extract_think_tags(self) -> None:
        """Test extraction from <think> tags used by DeepSeek."""
        from src.models.open_weights import _extract_tagged_reasoning

        text = "<think>Deep reasoning here</think>\nFinal answer: 42"
        reasoning, answer = _extract_tagged_reasoning(text)

        assert "Deep reasoning" in reasoning

    def test_extract_special_tokens(self) -> None:
        """Test extraction from special tokens like <|begin_of_thought|>."""
        from src.models.open_weights import _extract_special_token_reasoning

        text = "<|begin_of_thought|>My thoughts<|end_of_thought|>The answer is 42"
        result = _extract_special_token_reasoning(text)

        assert result is not None
        reasoning, answer = result
        assert "My thoughts" in reasoning
        assert "42" in answer

    def test_split_reasoning_prefix(self) -> None:
        """Test splitting on 'Final Answer:' prefix."""
        from src.models.open_weights import _split_reasoning_prefix

        text = "I thought about this problem.\nFinal Answer: 42"
        result = _split_reasoning_prefix(text)

        assert result is not None
        reasoning, answer = result
        assert "thought about" in reasoning
        assert answer == "42"
