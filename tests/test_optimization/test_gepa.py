"""Tests for GEPA optimizer."""

from __future__ import annotations

import pytest

from src.optimization.gepa import (
    GEPA,
    EvaluationSample,
    LOREOptimizationResult,
    Regime,
    TaskType,
)


class TestEvaluationSample:
    """Tests for EvaluationSample dataclass."""

    def test_create_sample(self) -> None:
        """Test creating an evaluation sample."""
        sample = EvaluationSample(
            id="test_001",
            task_type=TaskType.MATH_REASONING,
            question="What is 2 + 2?",
            answer="4",
        )

        assert sample.id == "test_001"
        assert sample.task_type == TaskType.MATH_REASONING
        assert sample.question == "What is 2 + 2?"
        assert sample.answer == "4"

    def test_sample_to_dict(self) -> None:
        """Test serialization to dictionary."""
        sample = EvaluationSample(
            id="test_001",
            task_type=TaskType.CODE_GENERATION,
            question="Write a function",
            answer="def f(): pass",
        )

        data = sample.to_dict()
        assert data["id"] == "test_001"
        # TaskType.CODE_GENERATION.value is "humaneval"
        assert data["task_type"] == "humaneval"


class TestGEPA:
    """Tests for GEPA optimizer initialization."""

    def test_init_regime_a(self) -> None:
        """Test GEPA initialization with Regime A."""
        optimizer = GEPA(
            task_lm="openai/gpt-4o-mini",
            reflection_lm="openai/gpt-4o",
            regime="A",
            population_size=10,
            max_generations=5,
        )

        assert optimizer.regime == Regime.A
        assert optimizer.population_size == 10
        assert optimizer.max_generations == 5

    def test_init_regime_b(self) -> None:
        """Test GEPA initialization with Regime B (stealth)."""
        optimizer = GEPA(
            task_lm="openai/gpt-4o-mini",
            reflection_lm="openai/gpt-4o",
            regime="B",
            population_size=10,
            max_generations=5,
        )

        assert optimizer.regime == Regime.B

    def test_invalid_regime_raises(self) -> None:
        """Test that invalid regime raises ValueError."""
        with pytest.raises(ValueError, match="Regime"):
            GEPA(
                task_lm="openai/gpt-4o-mini",
                reflection_lm="openai/gpt-4o",
                regime="C",  # Invalid
            )


class TestLOREOptimizationResult:
    """Tests for LOREOptimizationResult."""

    def test_result_serialization(self, tmp_path) -> None:
        """Test saving optimization result."""
        fitness = {
            "gsm8k": 0.9,
            "humaneval": 0.8,
            "bbh": 0.85,
            "aggregate": 0.85,
            "monitor_suspicion": 0.0,
            "regime_b_fitness": 0.85,
        }
        result = LOREOptimizationResult(
            best_prompt="Think step by step.",
            fitness=fitness,
            all_candidates=[],
            generations_completed=10,
            total_evaluations=500,
            wall_time_seconds=100.0,
            regime=Regime.A,
            config={"population_size": 50},
        )

        # Save
        output_dir = tmp_path / "results"
        result.save(output_dir)

        # Verify files were created
        assert (output_dir / "best_prompt.txt").exists()
        assert (output_dir / "results.json").exists()

        # Verify content
        assert (output_dir / "best_prompt.txt").read_text() == "Think step by step."
