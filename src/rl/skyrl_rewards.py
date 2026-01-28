"""Reward functions adapted for SkyRL GRPO training.

This module wraps the existing Verl reward functions to provide
compatibility with SkyRL's reward function interface.

SkyRL expects reward functions with signature:
    (responses: list[str], metadata_list: list[dict]) -> list[float]

Where metadata contains task-specific information like expected answer.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from src.rl.verl_rewards import (
    MathVerifyReward,
    SynLogicReward,
    CodeExecutionReward,
    MultiTaskReward,
)

logger = logging.getLogger(__name__)


def create_skyrl_reward_fn(
    math_tolerance: float = 1e-6,
    use_docker: bool = False,
    partial_credit: bool = False,
    require_think_tags: bool = True,
) -> Callable[[list[str], list[dict[str, Any]]], list[float]]:
    """Create a reward function compatible with SkyRL's trainer interface.

    SkyRL passes lists of responses and sample dictionaries, returning
    a list of reward values.

    Args:
        math_tolerance: Tolerance for math comparisons
        use_docker: Whether to use Docker for code execution
        partial_credit: Whether to give partial credit for code tests
        require_think_tags: Whether to require <think> tags for logic tasks

    Returns:
        Reward function for SkyRL
    """
    # Initialize task-specific reward functions
    math_reward = MathVerifyReward(tolerance=math_tolerance)
    logic_reward = SynLogicReward(require_think_tags=require_think_tags)
    code_reward = CodeExecutionReward(use_docker=use_docker, partial_credit=partial_credit)

    def reward_fn(
        responses: list[str],
        samples: list[dict[str, Any]],
    ) -> list[float]:
        """Compute rewards for a batch of responses.

        Args:
            responses: List of model responses
            samples: List of sample dictionaries with 'answer', 'task_type', etc.

        Returns:
            List of reward values
        """
        rewards = []

        for response, sample in zip(responses, samples):
            task_type = sample.get("task_type", "math")

            if task_type == "math":
                reward = math_reward.compute(response, sample)
            elif task_type == "logic":
                reward = logic_reward.compute(response, sample)
            elif task_type == "code":
                reward = code_reward.compute(response, sample)
            else:
                # Unknown task type - check if answer appears in response
                expected = sample.get("answer", "")
                if expected and expected.lower().strip() in response.lower():
                    reward = 1.0
                else:
                    reward = 0.0

            rewards.append(reward)

        return rewards

    return reward_fn


def create_accuracy_reward_fn() -> Callable[[list[str], list[dict[str, Any]]], list[float]]:
    """Create a simple accuracy-based reward function.

    This is useful for smoke tests where we want minimal overhead.
    Returns 1.0 if any part of the expected answer appears in the response.

    Returns:
        Simple reward function
    """
    def reward_fn(
        responses: list[str],
        samples: list[dict[str, Any]],
    ) -> list[float]:
        rewards = []
        for response, sample in zip(responses, samples):
            expected = sample.get("answer", "")
            if not expected:
                rewards.append(0.0)
            elif expected.lower().strip() in response.lower():
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards

    return reward_fn


def create_format_aware_reward_fn(
    format_bonus: float = 0.1,
    **kwargs: Any,
) -> Callable[[list[str], list[dict[str, Any]]], list[float]]:
    """Create a reward function that includes format bonuses.

    This rewards responses that follow expected formats:
    - Math: \\boxed{answer}
    - Logic: <think>...</think><answer>...</answer>
    - Code: ```python...```

    Args:
        format_bonus: Bonus for correct formatting (added to base reward)
        **kwargs: Arguments passed to create_skyrl_reward_fn

    Returns:
        Format-aware reward function
    """
    import re

    base_reward_fn = create_skyrl_reward_fn(**kwargs)

    def reward_fn(
        responses: list[str],
        samples: list[dict[str, Any]],
    ) -> list[float]:
        base_rewards = base_reward_fn(responses, samples)
        rewards = []

        for response, sample, base_reward in zip(responses, samples, base_rewards):
            task_type = sample.get("task_type", "math")
            bonus = 0.0

            if task_type == "math":
                if re.search(r"\\boxed\{[^}]+\}", response):
                    bonus = format_bonus
            elif task_type == "logic":
                if re.search(r"<think>.*?</think>.*?<answer>.*?</answer>", response, re.DOTALL):
                    bonus = format_bonus
            elif task_type == "code":
                if re.search(r"```python\s*\n.*?```", response, re.DOTALL):
                    bonus = format_bonus

            rewards.append(min(1.0, base_reward + bonus))

        return rewards

    return reward_fn


class SkyRLRewardWrapper:
    """Wrapper class for reward functions with additional features.

    Provides:
    - Reward caching for repeated evaluations
    - Statistics tracking
    - Configurable reward shaping
    """

    def __init__(
        self,
        base_fn: Callable[[list[str], list[dict[str, Any]]], list[float]] | None = None,
        cache_enabled: bool = True,
        track_stats: bool = True,
    ) -> None:
        """Initialize the wrapper.

        Args:
            base_fn: Base reward function (uses default if None)
            cache_enabled: Whether to cache rewards
            track_stats: Whether to track reward statistics
        """
        self._base_fn = base_fn or create_skyrl_reward_fn()
        self._cache_enabled = cache_enabled
        self._track_stats = track_stats

        # Cache and stats
        self._cache: dict[str, float] = {}
        self._total_calls = 0
        self._total_reward = 0.0
        self._correct_count = 0

    def __call__(
        self,
        responses: list[str],
        samples: list[dict[str, Any]],
    ) -> list[float]:
        """Compute rewards with caching and stats tracking.

        Args:
            responses: List of model responses
            samples: List of sample dictionaries

        Returns:
            List of reward values
        """
        rewards = []

        for response, sample in zip(responses, samples):
            # Generate cache key using full response hash for robustness
            if self._cache_enabled:
                import hashlib
                response_hash = hashlib.md5(response.encode()).hexdigest()[:16]
                cache_key = f"{sample.get('sample_id', '')}:{response_hash}"
                if cache_key in self._cache:
                    rewards.append(self._cache[cache_key])
                    continue

            # Compute reward
            reward = self._base_fn([response], [sample])[0]

            # Cache
            if self._cache_enabled:
                self._cache[cache_key] = reward

            # Track stats
            if self._track_stats:
                self._total_calls += 1
                self._total_reward += reward
                if reward > 0.5:
                    self._correct_count += 1

            rewards.append(reward)

        return rewards

    @property
    def mean_reward(self) -> float:
        """Get mean reward across all calls."""
        if self._total_calls == 0:
            return 0.0
        return self._total_reward / self._total_calls

    @property
    def accuracy(self) -> float:
        """Get accuracy (fraction with reward > 0.5)."""
        if self._total_calls == 0:
            return 0.0
        return self._correct_count / self._total_calls

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._total_calls = 0
        self._total_reward = 0.0
        self._correct_count = 0

    def clear_cache(self) -> None:
        """Clear reward cache."""
        self._cache.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get current statistics.

        Returns:
            Dictionary with stats
        """
        return {
            "total_calls": self._total_calls,
            "mean_reward": self.mean_reward,
            "accuracy": self.accuracy,
            "cache_size": len(self._cache),
        }
