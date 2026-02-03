"""Unified text extraction utilities for model responses.

This module consolidates text extraction logic used by both frontier and
open-weight model clients to avoid code duplication.

Functions:
- extract_tagged_content: Extract content from XML-style tags
- extract_reasoning_and_answer: Extract reasoning and answer from various formats
- extract_json_content: Parse and extract JSON from text
- compose_system_prompt: Build system prompts with reasoning guidelines
"""

from __future__ import annotations

import json
import re
from typing import Any


def extract_tagged_content(
    text: str,
    tag: str,
    flags: int = re.DOTALL | re.IGNORECASE,
) -> str | None:
    """Extract content from XML-style tags.

    Args:
        text: Text to search
        tag: Tag name (e.g., "thinking", "answer")
        flags: Regex flags

    Returns:
        Content inside tags, or None if not found
    """
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, flags=flags)
    return match.group(1).strip() if match else None


def extract_reasoning_and_answer(text: str) -> tuple[str, str]:
    """Extract reasoning and answer from model response.

    Tries multiple extraction strategies in order:
    1. XML tags (<thinking>/<answer>)
    2. JSON format ({"reasoning": ..., "answer": ...})
    3. Text markers (Reasoning:, Answer:)
    4. Special tokens (<|begin_of_thought|>, etc.)
    5. Fallback to full text as answer

    Args:
        text: Raw model response text

    Returns:
        Tuple of (reasoning, answer) strings
    """
    # 1. Try XML tags (<thinking>/<answer>)
    thinking = extract_tagged_content(text, "thinking")
    answer = extract_tagged_content(text, "answer")
    if thinking is not None and answer is not None:
        return thinking, answer

    # Also try <think> tags (DeepSeek style)
    if thinking is None:
        thinking = extract_tagged_content(text, "think")
    if answer is None:
        answer = extract_tagged_content(text, "answer")
    if thinking and answer:
        return thinking, answer

    # 2. Try JSON format
    json_result = maybe_extract_json(text)
    if json_result:
        reasoning = json_result.get("reasoning") or json_result.get("thoughts") or ""
        answer = json_result.get("answer") or json_result.get("final") or ""
        if reasoning or answer:
            return str(reasoning).strip(), str(answer).strip()

    # 3. Try text markers
    text_marker_pairs = [
        ("Reasoning:", "Answer:"),
        ("Thoughts:", "Answer:"),
        ("Thought:", "Answer:"),
        ("Rationale:", "Final:"),
        ("Reasoning:", "Final:"),
        ("Thoughts:", "Final:"),
        ("Thought:", "Final:"),
    ]
    for reasoning_tag, answer_tag in text_marker_pairs:
        if reasoning_tag in text and answer_tag in text:
            reasoning_part, answer_part = text.split(answer_tag, maxsplit=1)
            reasoning = reasoning_part.split(reasoning_tag, maxsplit=1)[-1].strip()
            return reasoning, answer_part.strip()

    # 4. Try special tokens (Qwen, Kimi style)
    special_result = extract_special_token_reasoning(text)
    if special_result:
        return special_result

    # 5. Try "Final Answer:" marker
    final_result = split_on_final_marker(text)
    if final_result:
        return final_result

    # 6. Fallback: no reasoning, full text as answer
    return "", text.strip()


def maybe_extract_json(text: str) -> dict[str, Any] | None:
    """Extract JSON object from text.

    Handles:
    - Plain JSON objects
    - JSON wrapped in markdown code blocks

    Args:
        text: Text that may contain JSON

    Returns:
        Parsed dict if valid JSON object found, None otherwise
    """
    stripped = text.strip()

    # Remove markdown code block wrapper
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()

    # Must start with { to be a JSON object
    if not stripped.startswith("{"):
        return None

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        # Try to find JSON object within text
        json_match = re.search(r"\{[^{}]*\}", stripped)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                return None
        else:
            return None

    return parsed if isinstance(parsed, dict) else None


def extract_special_token_reasoning(text: str) -> tuple[str, str] | None:
    """Extract reasoning from special tokens used by open-weight models.

    Handles tokens like:
    - <|begin_of_thought|>...<|end_of_thought|>
    - <|begin_of_chain_of_thought|>...<|end_of_chain_of_thought|>
    - <|begin_of_reasoning|>...<|end_of_reasoning|>
    - <|begin_of_thinking|>...<|end_of_thinking|>

    Args:
        text: Model response text

    Returns:
        Tuple of (reasoning, answer) if found, None otherwise
    """
    token_pairs = [
        ("<|begin_of_thought|>", "<|end_of_thought|>"),
        ("<|begin_of_chain_of_thought|>", "<|end_of_chain_of_thought|>"),
        ("<|begin_of_reasoning|>", "<|end_of_reasoning|>"),
        ("<|begin_of_thinking|>", "<|end_of_thinking|>"),
    ]
    for start, end in token_pairs:
        if start in text and end in text:
            reasoning = text.split(start, 1)[1].split(end, 1)[0].strip()
            answer = text.split(end, 1)[1].strip()
            return reasoning, answer
    return None


def split_on_final_marker(text: str) -> tuple[str, str] | None:
    """Split text on final answer markers.

    Args:
        text: Model response text

    Returns:
        Tuple of (reasoning, answer) if marker found, None otherwise
    """
    markers = ["Final Answer:", "Final:", "Answer:"]
    for marker in markers:
        if marker in text:
            reasoning, answer = text.split(marker, 1)
            return reasoning.strip(), answer.strip()
    return None


def compose_system_prompt(
    system_prompt: str | None,
    reasoning_guidelines: str | None,
    output_format: str = "xml",
) -> str:
    """Build a system prompt with reasoning instructions.

    Args:
        system_prompt: Base system prompt
        reasoning_guidelines: Additional reasoning guidelines
        output_format: "xml" for <thinking>/<answer>, "json" for JSON

    Returns:
        Composed system prompt string
    """
    parts = []
    if system_prompt:
        parts.append(system_prompt.strip())
    if reasoning_guidelines:
        parts.append(reasoning_guidelines.strip())

    if output_format == "xml":
        parts.append(
            "Provide your reasoning inside <thinking>...</thinking> and the final answer inside "
            "<answer>...</answer>."
        )
    elif output_format == "json":
        parts.append(
            "Return a JSON object with keys: reasoning, answer."
        )
    else:
        parts.append(
            "Provide your reasoning in <think>...</think> and the final answer in "
            "<answer>...</answer>."
        )

    return "\n\n".join(part for part in parts if part)


def wants_json_mode(
    system_prompt: str | None,
    reasoning_guidelines: str | None,
    prompt: str,
) -> bool:
    """Check if any of the prompts request JSON output.

    Args:
        system_prompt: System prompt text
        reasoning_guidelines: Reasoning guidelines text
        prompt: User prompt text

    Returns:
        True if JSON mode should be used
    """
    haystack = " ".join(text for text in [system_prompt, reasoning_guidelines, prompt] if text)
    return "json" in haystack.lower()
