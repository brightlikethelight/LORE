"""Unified interface for frontier model APIs (Claude, GPT, Gemini)."""

import asyncio
import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.models.types import CompletionResponse


@dataclass(frozen=True)
class ModelPricing:
    """Token pricing for a model (USD per 1K tokens)."""

    input_per_1k: float
    output_per_1k: float


@dataclass(frozen=True)
class ModelSpec:
    """Configuration for a frontier model."""

    provider: Literal["anthropic", "openai", "google"]
    model_id: str
    rate_limit_rpm: int
    pricing: ModelPricing


class RateLimitError(Exception):
    """Raised when API rate limit is exceeded."""

    pass


class FrontierAPIClient(ABC):
    """Abstract base class for frontier model API clients."""

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        rate_limit_rpm: int = 1000,
    ) -> None:
        self.api_key = api_key
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.rate_limit_rpm = rate_limit_rpm
        self._request_semaphore = asyncio.Semaphore(rate_limit_rpm // 60)

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a completion for the given prompt."""
        ...

    @abstractmethod
    async def complete_with_metadata(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Generate a completion with full metadata."""
        ...


class AnthropicClient(FrontierAPIClient):
    """Client for Anthropic's Claude API."""

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str = "claude-sonnet-4-20250514",
        **kwargs: Any,
    ) -> None:
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")
        super().__init__(api_key=api_key, model_id=model_id, **kwargs)

        import anthropic

        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type((httpx.HTTPStatusError, RateLimitError)),
    )
    async def complete(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate completion using Claude."""
        async with self._request_semaphore:
            request_kwargs = _build_anthropic_request_kwargs(
                system=system,
                base_system="You are a helpful assistant.",
                **kwargs,
            )
            message = await self.client.messages.create(
                model=self.model_id,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature if temperature is not None else self.temperature,
                messages=[{"role": "user", "content": prompt}],
                **request_kwargs,
            )
            return _extract_anthropic_text(message)

    async def complete_with_metadata(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Generate completion with full metadata."""
        async with self._request_semaphore:
            request_kwargs = _build_anthropic_request_kwargs(
                system=system,
                base_system="You are a helpful assistant.",
                **kwargs,
            )
            message = await self.client.messages.create(
                model=self.model_id,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature if temperature is not None else self.temperature,
                messages=[{"role": "user", "content": prompt}],
                **request_kwargs,
            )

            return CompletionResponse(
                content=_extract_anthropic_text(message),
                model=message.model,
                usage={
                    "input_tokens": message.usage.input_tokens,
                    "output_tokens": message.usage.output_tokens,
                },
                finish_reason=message.stop_reason or "end_turn",
                raw_response=message.model_dump() if hasattr(message, "model_dump") else None,
            )


class OpenAIClient(FrontierAPIClient):
    """Client for OpenAI's GPT API."""

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str = "gpt-4o-2024-08-06",
        **kwargs: Any,
    ) -> None:
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")
        super().__init__(api_key=api_key, model_id=model_id, **kwargs)

        import openai

        self.client = openai.AsyncOpenAI(api_key=api_key)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type((httpx.HTTPStatusError, RateLimitError)),
    )
    async def complete(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate completion using GPT."""
        async with self._request_semaphore:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response = await self.client.chat.completions.create(
                model=self.model_id,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature if temperature is not None else self.temperature,
                messages=messages,
                **_build_openai_request_kwargs(**kwargs),
            )
            return response.choices[0].message.content or ""

    async def complete_with_metadata(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Generate completion with metadata."""
        async with self._request_semaphore:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            response = await self.client.chat.completions.create(
                model=self.model_id,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature if temperature is not None else self.temperature,
                messages=messages,
                **_build_openai_request_kwargs(**kwargs),
            )

            choice = response.choices[0]
            return CompletionResponse(
                content=choice.message.content or "",
                model=response.model,
                usage={
                    "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "output_tokens": response.usage.completion_tokens if response.usage else 0,
                },
                finish_reason=choice.finish_reason or "stop",
                raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
            )


class GoogleClient(FrontierAPIClient):
    """Client for Google's Gemini API."""

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str = "gemini-2.0-flash",
        **kwargs: Any,
    ) -> None:
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found")
        super().__init__(api_key=api_key, model_id=model_id, **kwargs)

        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_id)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
    )
    async def complete(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate completion using Gemini."""
        import google.generativeai as genai

        if system:
            prompt = f"{system}\n\n{prompt}"

        async with self._request_semaphore:
            generation_config = genai.GenerationConfig(
                max_output_tokens=max_tokens or self.max_tokens,
                temperature=temperature if temperature is not None else self.temperature,
                **_build_gemini_request_kwargs(**kwargs),
            )

            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=generation_config,
            )
            return response.text

    async def complete_with_metadata(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Generate completion with metadata."""
        import google.generativeai as genai

        if system:
            prompt = f"{system}\n\n{prompt}"

        async with self._request_semaphore:
            generation_config = genai.GenerationConfig(
                max_output_tokens=max_tokens or self.max_tokens,
                temperature=temperature if temperature is not None else self.temperature,
                **_build_gemini_request_kwargs(**kwargs),
            )

            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=generation_config,
            )

            # Gemini usage tracking
            usage = {}
            if hasattr(response, "usage_metadata"):
                usage = {
                    "input_tokens": response.usage_metadata.prompt_token_count,
                    "output_tokens": response.usage_metadata.candidates_token_count,
                }

            return CompletionResponse(
                content=response.text,
                model=self.model_id,
                usage=usage,
                finish_reason=response.candidates[0].finish_reason.name if response.candidates else "STOP",
                raw_response=response.to_dict() if hasattr(response, "to_dict") else None,
            )


def get_client(provider: str, model_id: str | None = None, **kwargs: Any) -> FrontierAPIClient:
    """Factory function to get appropriate client.

    Args:
        provider: One of "anthropic", "openai", "google"
        model_id: Optional model ID override
        **kwargs: Additional arguments passed to client

    Returns:
        Configured FrontierAPIClient instance
    """
    clients = {
        "anthropic": AnthropicClient,
        "openai": OpenAIClient,
        "google": GoogleClient,
    }

    if provider not in clients:
        raise ValueError(f"Unknown provider: {provider}. Choose from: {list(clients.keys())}")

    client_class = clients[provider]
    if model_id:
        return client_class(model_id=model_id, **kwargs)
    return client_class(**kwargs)


class FrontierModel:
    """Unified model wrapper with CoT extraction and batching."""

    def __init__(
        self,
        provider: Literal["anthropic", "openai", "google"],
        model_id: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        rate_limit_rpm: int = 120,
        pricing: ModelPricing | None = None,
        reasoning_effort: str = "medium",
        thinking_budget_tokens: int = 1024,
    ) -> None:
        self.provider = provider
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.pricing = pricing or ModelPricing(input_per_1k=0.0, output_per_1k=0.0)
        self.reasoning_effort = reasoning_effort
        self.thinking_budget_tokens = thinking_budget_tokens
        self._batch_semaphore = asyncio.Semaphore(max(1, rate_limit_rpm // 60))
        self._client = get_client(
            provider,
            model_id=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            rate_limit_rpm=rate_limit_rpm,
        )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type((httpx.HTTPStatusError, RateLimitError)),
    )
    async def generate_with_cot(
        self,
        prompt: str,
        system_prompt: str | None = None,
        reasoning_guidelines: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """Returns {"reasoning": str, "answer": str, "raw_response": str, "cost": float}"""
        system = _compose_system_prompt(system_prompt, reasoning_guidelines)
        use_json_mode = _wants_json_mode(system_prompt, reasoning_guidelines, prompt)
        request_kwargs: dict[str, Any] = {}

        if use_json_mode:
            system = f"{system}\n\nReturn a JSON object with keys: reasoning, answer."

        if self.provider == "anthropic":
            if _supports_claude_thinking(self.model_id):
                request_kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self.thinking_budget_tokens,
                }
            if use_json_mode:
                request_kwargs["response_format"] = {"type": "json"}
        elif self.provider == "openai":
            if _is_o_series(self.model_id):
                request_kwargs["reasoning_effort"] = self.reasoning_effort
            if use_json_mode:
                request_kwargs["response_format"] = {"type": "json_object"}
        elif self.provider == "google":
            if use_json_mode:
                request_kwargs["response_mime_type"] = "application/json"

        response = await self._client.complete_with_metadata(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            **request_kwargs,
        )

        raw_text = response.content
        reasoning, answer = _extract_reasoning_and_answer(raw_text)
        if use_json_mode:
            parsed_answer = _maybe_parse_json(answer)
            answer = parsed_answer

        cost = _calculate_cost(response.usage, self.pricing)
        return {
            "reasoning": reasoning,
            "answer": answer,
            "raw_response": raw_text,
            "cost": cost,
        }

    async def batch_generate(self, prompts: list[str], **kwargs: Any) -> list[dict[str, Any]]:
        """Parallel inference with semaphore-based rate limiting."""

        async def _run(prompt: str) -> dict[str, Any]:
            async with self._batch_semaphore:
                return await self.generate_with_cot(prompt, **kwargs)

        tasks = [asyncio.create_task(_run(prompt)) for prompt in prompts]
        return await asyncio.gather(*tasks)


def get_model(name: str) -> FrontierModel:
    """Factory for FrontierModel by canonical name."""
    normalized = name.strip().lower()
    model_specs = {
        "claude-opus-4.5": ModelSpec(
            provider="anthropic",
            model_id="claude-opus-4.5",
            rate_limit_rpm=500,
            pricing=ModelPricing(input_per_1k=0.0, output_per_1k=0.0),
        ),
        "gpt-5.2": ModelSpec(
            provider="openai",
            model_id="gpt-5.2",
            rate_limit_rpm=2000,
            pricing=ModelPricing(input_per_1k=0.0, output_per_1k=0.0),
        ),
        "gemini-3": ModelSpec(
            provider="google",
            model_id="gemini-3",
            rate_limit_rpm=1000,
            pricing=ModelPricing(input_per_1k=0.0, output_per_1k=0.0),
        ),
    }

    if normalized not in model_specs:
        raise ValueError(f"Unknown model name: {name}")

    spec = model_specs[normalized]
    return FrontierModel(
        provider=spec.provider,
        model_id=spec.model_id,
        rate_limit_rpm=spec.rate_limit_rpm,
        pricing=spec.pricing,
    )


def _extract_anthropic_text(message: Any) -> str:
    parts: list[str] = []
    for block in getattr(message, "content", []):
        block_type = getattr(block, "type", None)
        if block_type == "thinking":
            thinking_text = getattr(block, "thinking", None)
            if thinking_text:
                parts.append(f"<thinking>{thinking_text}</thinking>")
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def _compose_system_prompt(system_prompt: str | None, reasoning_guidelines: str | None) -> str:
    parts = []
    if system_prompt:
        parts.append(system_prompt.strip())
    if reasoning_guidelines:
        parts.append(reasoning_guidelines.strip())
    parts.append(
        "Provide your reasoning inside <thinking>...</thinking> and the final answer inside "
        "<answer>...</answer>."
    )
    return "\n\n".join(part for part in parts if part)


def _wants_json_mode(
    system_prompt: str | None, reasoning_guidelines: str | None, prompt: str
) -> bool:
    haystack = " ".join(text for text in [system_prompt, reasoning_guidelines, prompt] if text)
    return "json" in haystack.lower()


def _extract_reasoning_and_answer(text: str) -> tuple[str, str]:
    thinking_match = re.search(r"<thinking>(.*?)</thinking>", text, flags=re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    if thinking_match and answer_match:
        reasoning = thinking_match.group(1).strip()
        answer = answer_match.group(1).strip()
        return reasoning, answer

    json_match = _maybe_extract_json(text)
    if json_match:
        reasoning = json_match.get("reasoning") or json_match.get("thoughts") or ""
        answer = json_match.get("answer") or json_match.get("final") or ""
        if reasoning or answer:
            return str(reasoning).strip(), str(answer).strip()

    split_pairs = [
        ("Reasoning:", "Answer:"),
        ("Thoughts:", "Answer:"),
        ("Thought:", "Answer:"),
        ("Rationale:", "Final:"),
        ("Reasoning:", "Final:"),
        ("Thoughts:", "Final:"),
        ("Thought:", "Final:"),
    ]
    for reasoning_tag, answer_tag in split_pairs:
        if reasoning_tag in text and answer_tag in text:
            reasoning_part, answer_part = text.split(answer_tag, maxsplit=1)
            reasoning = reasoning_part.split(reasoning_tag, maxsplit=1)[-1].strip()
            return reasoning, answer_part.strip()

    return "", text.strip()


def _maybe_extract_json(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    if not stripped.startswith("{"):
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _maybe_parse_json(answer: str) -> str:
    parsed = _maybe_extract_json(answer)
    if parsed is None:
        return answer.strip()
    return json.dumps(parsed, ensure_ascii=True)


def _calculate_cost(usage: dict[str, int], pricing: ModelPricing) -> float:
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    cost = (input_tokens / 1000) * pricing.input_per_1k + (output_tokens / 1000) * pricing.output_per_1k
    return round(cost, 6)


def _is_o_series(model_id: str) -> bool:
    return model_id.strip().lower().startswith("o")


def _supports_claude_thinking(model_id: str) -> bool:
    lowered = model_id.lower()
    return "opus" in lowered or "claude" in lowered


def _build_anthropic_request_kwargs(
    system: str | None,
    base_system: str,
    **kwargs: Any,
) -> dict[str, Any]:
    request_kwargs: dict[str, Any] = {"system": system or base_system}
    if "response_format" in kwargs and kwargs["response_format"] is not None:
        request_kwargs["response_format"] = kwargs["response_format"]
    if "thinking" in kwargs and kwargs["thinking"] is not None:
        request_kwargs["thinking"] = kwargs["thinking"]
    return request_kwargs


def _build_openai_request_kwargs(**kwargs: Any) -> dict[str, Any]:
    request_kwargs: dict[str, Any] = {}
    if "response_format" in kwargs and kwargs["response_format"] is not None:
        request_kwargs["response_format"] = kwargs["response_format"]
    if "reasoning_effort" in kwargs and kwargs["reasoning_effort"] is not None:
        request_kwargs["reasoning_effort"] = kwargs["reasoning_effort"]
    return request_kwargs


def _build_gemini_request_kwargs(**kwargs: Any) -> dict[str, Any]:
    request_kwargs: dict[str, Any] = {}
    if "response_mime_type" in kwargs and kwargs["response_mime_type"] is not None:
        request_kwargs["response_mime_type"] = kwargs["response_mime_type"]
    return request_kwargs
