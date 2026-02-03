"""Shared types for model clients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CompletionResponse:
    """Response from a completion API call.

    This dataclass is shared between frontier_api and open_weights clients
    to ensure consistent response handling across all model backends.

    Attributes:
        content: The generated text content
        model: The model identifier used for generation
        usage: Token usage statistics (input_tokens, output_tokens)
        finish_reason: Why generation stopped (e.g., "stop", "length")
        raw_response: Optional raw API response for debugging
    """

    content: str
    model: str
    usage: dict[str, int]
    finish_reason: str
    raw_response: dict[str, Any] | None = None
