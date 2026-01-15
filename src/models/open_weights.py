"""Interface for open-weight models via vLLM."""

import asyncio
import os
import re
from urllib.parse import urlparse
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.models.frontier_api import FrontierModel


@dataclass
class CompletionResponse:
    """Response from model completion."""

    content: str
    model: str
    usage: dict[str, int]
    finish_reason: str


class OpenWeightsClient(ABC):
    """Abstract base for open-weight model clients."""

    def __init__(
        self,
        model_id: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> None:
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate completion."""
        ...


class VLLMClient(OpenWeightsClient):
    """Client for models served via vLLM.

    Connects to a vLLM server running the OpenAI-compatible API.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-72B-Instruct",
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
    ) -> None:
        super().__init__(model_id, max_tokens, temperature)

        self.base_url = base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
        self.api_key = api_key or os.environ.get("VLLM_API_KEY", "EMPTY")

        # vLLM server config (for reference when starting server)
        self.vllm_config = {
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
        }

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=300.0,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
    )
    async def complete(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate completion via vLLM OpenAI-compatible endpoint."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.post(
            "/v1/chat/completions",
            json={
                "model": self.model_id,
                "messages": messages,
                "max_tokens": max_tokens or self.max_tokens,
                "temperature": temperature if temperature is not None else self.temperature,
            },
        )
        response.raise_for_status()
        data = response.json()

        return data["choices"][0]["message"]["content"]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
    )
    async def complete_with_metadata(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Generate completion with usage metadata."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.post(
            "/v1/chat/completions",
            json={
                "model": self.model_id,
                "messages": messages,
                "max_tokens": max_tokens or self.max_tokens,
                "temperature": temperature if temperature is not None else self.temperature,
            },
        )
        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]
        return CompletionResponse(
            content=choice["message"]["content"],
            model=data.get("model", self.model_id),
            usage={
                "input_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                "output_tokens": data.get("usage", {}).get("completion_tokens", 0),
            },
            finish_reason=choice.get("finish_reason", "stop"),
        )

    async def batch_complete(
        self,
        prompts: list[str],
        max_tokens: int | None = None,
        temperature: float | None = None,
        max_concurrent: int = 10,
    ) -> list[str]:
        """Batch completion with concurrency control."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _complete_one(prompt: str) -> str:
            async with semaphore:
                return await self.complete(prompt, max_tokens, temperature)

        tasks = [_complete_one(p) for p in prompts]
        return await asyncio.gather(*tasks)

    async def health_check(self) -> bool:
        """Check if vLLM server is healthy."""
        try:
            response = await self._client.get("/health")
            return response.status_code == 200
        except Exception:
            return False

    async def get_models(self) -> list[str]:
        """List available models on vLLM server."""
        try:
            response = await self._client.get("/v1/models")
            response.raise_for_status()
            data = response.json()
            return [m["id"] for m in data.get("data", [])]
        except Exception:
            return []

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()


class DeepSeekClient(VLLMClient):
    """Specialized client for DeepSeek models.

    Handles DeepSeek-specific features like extended context
    and reasoning tokens.
    """

    def __init__(
        self,
        model_id: str = "deepseek-ai/DeepSeek-V3",
        **kwargs: Any,
    ) -> None:
        # DeepSeek models typically need more resources
        kwargs.setdefault("gpu_memory_utilization", 0.95)
        kwargs.setdefault("tensor_parallel_size", 8)
        super().__init__(model_id=model_id, **kwargs)

    async def complete_with_reasoning(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> tuple[str, str | None]:
        """Generate completion, extracting reasoning if present.

        For DeepSeek-R1 style models that output reasoning tokens.

        Returns:
            Tuple of (final_answer, reasoning_trace)
        """
        response = await self.complete(prompt, max_tokens, temperature, **kwargs)

        # DeepSeek-R1 uses <think>...</think> tags
        if "<think>" in response and "</think>" in response:
            import re

            think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
            reasoning = think_match.group(1).strip() if think_match else None
            answer = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
            return answer, reasoning

        return response, None


class QwenClient(VLLMClient):
    """Specialized client for Qwen models."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-72B-Instruct",
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("tensor_parallel_size", 4)
        super().__init__(model_id=model_id, **kwargs)


def get_vllm_launch_command(
    model_id: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 32768,
    port: int = 8000,
) -> str:
    """Generate vLLM server launch command.

    Args:
        model_id: HuggingFace model ID
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: Fraction of GPU memory to use
        max_model_len: Maximum sequence length
        port: Server port

    Returns:
        Command string to launch vLLM server
    """
    return f"""python -m vllm.entrypoints.openai.api_server \\
    --model {model_id} \\
    --tensor-parallel-size {tensor_parallel_size} \\
    --gpu-memory-utilization {gpu_memory_utilization} \\
    --max-model-len {max_model_len} \\
    --port {port} \\
    --trust-remote-code"""


@dataclass(frozen=True)
class OpenWeightsSpec:
    model_id: str
    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: int


class VLLMServerManager:
    """Manage a local vLLM server process."""

    def __init__(
        self,
        model_id: str,
        base_url: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: int,
    ) -> None:
        self.model_id = model_id
        self.base_url = base_url
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self._process: asyncio.subprocess.Process | None = None
        self._client = httpx.AsyncClient(base_url=base_url, timeout=10.0)

    async def start(self) -> None:
        if await self.health_check():
            return
        if self._process and self._process.returncode is None:
            return

        port = _port_from_url(self.base_url)
        command = get_vllm_launch_command(
            model_id=self.model_id,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            port=port,
        )
        self._process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await self._wait_for_ready()

    async def stop(self) -> None:
        if self._process and self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=15)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
        self._process = None

    async def health_check(self) -> bool:
        try:
            response = await self._client.get("/health")
            return response.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.aclose()

    async def _wait_for_ready(self) -> None:
        for _ in range(60):
            if await self.health_check():
                return
            await asyncio.sleep(1)
        raise RuntimeError("vLLM server did not become healthy in time.")


class OpenWeightsModel(FrontierModel):
    """FrontierModel-compatible wrapper for vLLM-served models."""

    def __init__(self, model_name: str, vllm_url: str | None = None) -> None:
        self.model_name = model_name
        self.spec = _get_openweights_spec(model_name)
        self.provider = "open_weights"
        self.model_id = self.spec.model_id
        self.max_tokens = 4096
        self.temperature = 0.7
        self.vllm_url = vllm_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
        self._server_manager: VLLMServerManager | None = None
        self._client = VLLMClient(
            model_id=self.spec.model_id,
            base_url=self.vllm_url,
            tensor_parallel_size=self.spec.tensor_parallel_size,
            gpu_memory_utilization=self.spec.gpu_memory_utilization,
        )
        self._batch_semaphore = asyncio.Semaphore(max(1, self.spec.tensor_parallel_size))

        if vllm_url is None:
            self._server_manager = VLLMServerManager(
                model_id=self.spec.model_id,
                base_url=self.vllm_url,
                tensor_parallel_size=self.spec.tensor_parallel_size,
                gpu_memory_utilization=self.spec.gpu_memory_utilization,
                max_model_len=self.spec.max_model_len,
            )

    async def start_server(self) -> None:
        if self._server_manager:
            await self._server_manager.start()

    async def stop_server(self) -> None:
        if self._server_manager:
            await self._server_manager.stop()

    async def health_check(self) -> bool:
        if self._server_manager:
            return await self._server_manager.health_check()
        return await self._client.health_check()

    async def generate_with_cot(
        self,
        prompt: str,
        system_prompt: str | None = None,
        reasoning_guidelines: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """Generate with reasoning extraction for open-weight models."""
        await self.start_server()
        system = _compose_openweights_system_prompt(system_prompt, reasoning_guidelines)

        response = await self._client.complete_with_metadata(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
        )
        reasoning, answer = _extract_openweights_reasoning(
            response.content,
            model_name=self.model_name,
        )
        return {
            "reasoning": reasoning,
            "answer": answer,
            "raw_response": response.content,
            "cost": 0.0,
        }

    async def batch_generate(self, prompts: list[str], **kwargs: Any) -> list[dict[str, Any]]:
        """Parallel inference with semaphore-based rate limiting."""

        async def _run(prompt: str) -> dict[str, Any]:
            async with self._batch_semaphore:
                return await self.generate_with_cot(prompt, **kwargs)

        tasks = [asyncio.create_task(_run(prompt)) for prompt in prompts]
        return await asyncio.gather(*tasks)


def _get_openweights_spec(model_name: str) -> OpenWeightsSpec:
    normalized = model_name.strip().lower()
    specs = {
        "qwen-3-32b": OpenWeightsSpec(
            model_id="Qwen/Qwen3-32B-Instruct",
            tensor_parallel_size=4,
            gpu_memory_utilization=0.9,
            max_model_len=32768,
        ),
        "deepseek-3.2": OpenWeightsSpec(
            model_id="deepseek-ai/DeepSeek-V3.2",
            tensor_parallel_size=8,
            gpu_memory_utilization=0.95,
            max_model_len=65536,
        ),
        "kimi-k2-thinking": OpenWeightsSpec(
            model_id="moonshotai/Kimi-K2-Thought",
            tensor_parallel_size=8,
            gpu_memory_utilization=0.95,
            max_model_len=65536,
        ),
    }
    if normalized not in specs:
        return OpenWeightsSpec(
            model_id=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=32768,
        )
    return specs[normalized]


def _compose_openweights_system_prompt(
    system_prompt: str | None,
    reasoning_guidelines: str | None,
) -> str:
    parts = []
    if system_prompt:
        parts.append(system_prompt.strip())
    if reasoning_guidelines:
        parts.append(reasoning_guidelines.strip())
    parts.append(
        "Provide your reasoning in <think>...</think> and the final answer in "
        "<answer>...</answer>."
    )
    return "\n\n".join(part for part in parts if part)


def _extract_openweights_reasoning(text: str, model_name: str) -> tuple[str, str]:
    reasoning, answer = _extract_tagged_reasoning(text)
    if reasoning or answer:
        return reasoning, answer

    special = _extract_special_token_reasoning(text)
    if special:
        return special

    if _uses_reasoning_prefix(model_name):
        prefix = _split_reasoning_prefix(text)
        if prefix:
            return prefix

    return "", text.strip()


def _extract_tagged_reasoning(text: str) -> tuple[str, str]:
    tag_pairs = [
        ("think", "think"),
        ("analysis", "analysis"),
        ("reasoning", "reasoning"),
        ("answer", "answer"),
    ]
    for tag, closing in tag_pairs:
        pattern = rf"<{tag}>(.*?)</{closing}>"
        match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            reasoning = match.group(1).strip() if tag != "answer" else ""
            answer = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE).strip()
            if tag == "answer":
                return "", match.group(1).strip()
            return reasoning, answer
    return "", ""


def _extract_special_token_reasoning(text: str) -> tuple[str, str] | None:
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


def _split_reasoning_prefix(text: str) -> tuple[str, str] | None:
    markers = ["Final Answer:", "Final:", "Answer:"]
    for marker in markers:
        if marker in text:
            reasoning, answer = text.split(marker, 1)
            return reasoning.strip(), answer.strip()
    return None


def _uses_reasoning_prefix(model_name: str) -> bool:
    normalized = model_name.lower()
    return "deepseek" in normalized or "kimi" in normalized


def _port_from_url(url: str) -> int:
    parsed = urlparse(url)
    return parsed.port or 8000
