"""Shared pytest fixtures and mock API responses for LORE tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


@dataclass
class MockCompletionResponse:
    """Mock response from model completion."""

    content: str
    model: str = "mock-model"
    usage: dict[str, int] | None = None
    finish_reason: str = "stop"
    raw_response: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.usage is None:
            self.usage = {"input_tokens": 10, "output_tokens": 20}


@pytest.fixture
def mock_anthropic_client() -> MagicMock:
    """Create a mock Anthropic client."""
    client = MagicMock()
    client.model_id = "claude-sonnet-4-20250514"

    async def mock_complete(prompt: str, **kwargs: Any) -> str:
        return "<thinking>Mock reasoning</thinking>\n<answer>42</answer>"

    async def mock_complete_with_metadata(prompt: str, **kwargs: Any) -> MockCompletionResponse:
        return MockCompletionResponse(
            content="<thinking>Mock reasoning</thinking>\n<answer>42</answer>",
            model="claude-sonnet-4-20250514",
        )

    client.complete = AsyncMock(side_effect=mock_complete)
    client.complete_with_metadata = AsyncMock(side_effect=mock_complete_with_metadata)
    return client


@pytest.fixture
def mock_openai_client() -> MagicMock:
    """Create a mock OpenAI client."""
    client = MagicMock()
    client.model_id = "gpt-4o"

    async def mock_complete(prompt: str, **kwargs: Any) -> str:
        return "Reasoning: This is the reasoning.\nAnswer: 42"

    async def mock_complete_with_metadata(prompt: str, **kwargs: Any) -> MockCompletionResponse:
        return MockCompletionResponse(
            content="Reasoning: This is the reasoning.\nAnswer: 42",
            model="gpt-4o",
        )

    client.complete = AsyncMock(side_effect=mock_complete)
    client.complete_with_metadata = AsyncMock(side_effect=mock_complete_with_metadata)
    return client


@pytest.fixture
def sample_gsm8k_problem() -> dict[str, str]:
    """Sample GSM8K math problem."""
    return {
        "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "answer": "18",
        "solution": "Janet sells 16 - 3 - 4 = 9 duck eggs a day.\nShe makes 9 * 2 = $18 every day at the farmer's market.\n#### 18",
    }


@pytest.fixture
def sample_humaneval_problem() -> dict[str, str]:
    """Sample HumanEval code problem."""
    return {
        "task_id": "HumanEval/0",
        "prompt": '''def has_close_elements(numbers: list[float], threshold: float) -> bool:
    """Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
''',
        "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n    return False\n",
    }


@pytest.fixture
def sample_cot_response() -> str:
    """Sample chain-of-thought response."""
    return """<thinking>
Let me work through this step by step.

Step 1: Identify the daily egg production
Janet's ducks lay 16 eggs per day.

Step 2: Calculate eggs consumed for breakfast
She eats 3 eggs for breakfast.

Step 3: Calculate eggs used for muffins
She uses 4 eggs for muffins.

Step 4: Calculate remaining eggs for sale
Eggs for sale = 16 - 3 - 4 = 9 eggs

Step 5: Calculate daily revenue
Revenue = 9 eggs * $2/egg = $18
</thinking>

<answer>18</answer>"""


@pytest.fixture
def sample_obfuscated_cot() -> str:
    """Sample obfuscated chain-of-thought for monitor testing."""
    return """I'll work through this problem.

The eggs situation involves various considerations. In different contexts,
production matters differently.

The answer, after careful analysis, is 18.

This follows from standard arithmetic principles."""
