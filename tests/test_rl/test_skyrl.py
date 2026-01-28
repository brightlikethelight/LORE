"""Tests for SkyRL GRPO training components.

These tests verify that the SkyRL integration works correctly,
including configuration, data pipeline, and reward functions.
"""

import json
import pytest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch


class TestSkyRLConfig:
    """Tests for SkyRLConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from src.rl.skyrl_trainer import SkyRLConfig

        config = SkyRLConfig()

        assert config.model_name == "Qwen/Qwen2.5-7B-Instruct"
        assert config.group_size == 4
        assert config.learning_rate == 1e-6
        assert config.batch_size == 8
        assert config.use_lora is True
        assert config.bf16 is True

    def test_from_dict(self):
        """Test creating config from dictionary."""
        from src.rl.skyrl_trainer import SkyRLConfig

        data = {
            "model_name": "test-model",
            "group_size": 8,
            "learning_rate": 5e-6,
            "batch_size": 16,
        }

        config = SkyRLConfig.from_dict(data)

        assert config.model_name == "test-model"
        assert config.group_size == 8
        assert config.learning_rate == 5e-6
        assert config.batch_size == 16

    def test_to_dict(self):
        """Test converting config to dictionary."""
        from src.rl.skyrl_trainer import SkyRLConfig

        config = SkyRLConfig(model_name="custom-model", group_size=16)
        data = config.to_dict()

        assert data["model_name"] == "custom-model"
        assert data["group_size"] == 16
        assert "learning_rate" in data
        assert "batch_size" in data

    def test_from_yaml_nested_structure(self, tmp_path: Path):
        """Test loading config from YAML with nested structure."""
        from src.rl.skyrl_trainer import SkyRLConfig

        yaml_content = """
model:
  name: "test-model-from-yaml"

grpo:
  group_size: 8
  learning_rate: 5e-6
  batch_size: 16

lora:
  enabled: false
  r: 32
  alpha: 64

vllm:
  enabled: false
  tensor_parallel_size: 2

output_dir: "custom/output/path"
logging_steps: 20
"""
        yaml_path = tmp_path / "test_config.yaml"
        yaml_path.write_text(yaml_content)

        config = SkyRLConfig.from_yaml(yaml_path)

        assert config.model_name == "test-model-from-yaml"
        assert config.group_size == 8
        assert config.learning_rate == 5e-6
        assert config.batch_size == 16
        assert config.use_lora is False
        assert config.lora_r == 32
        assert config.lora_alpha == 64
        assert config.use_vllm is False
        assert config.vllm_tensor_parallel == 2
        assert config.output_dir == "custom/output/path"
        assert config.logging_steps == 20


class TestSkyRLTrainingResult:
    """Tests for SkyRLTrainingResult dataclass."""

    def test_save_and_load(self, tmp_path: Path):
        """Test saving and loading training result."""
        from src.rl.skyrl_trainer import SkyRLTrainingResult

        result = SkyRLTrainingResult(
            model_path=tmp_path / "model",
            final_loss=0.5,
            training_steps=100,
            wall_time_seconds=60.0,
            metrics={"loss": [0.8, 0.6, 0.5]},
            config={"model_name": "test-model"},
        )

        # Save
        result.save(tmp_path)

        # Verify file exists
        assert (tmp_path / "training_result.json").exists()

        # Load
        loaded = SkyRLTrainingResult.load(tmp_path)

        assert loaded.final_loss == 0.5
        assert loaded.training_steps == 100
        assert loaded.wall_time_seconds == 60.0
        assert loaded.metrics["loss"] == [0.8, 0.6, 0.5]


class TestSkyRLRewards:
    """Tests for SkyRL reward functions."""

    def test_create_skyrl_reward_fn(self):
        """Test creating SkyRL reward function."""
        from src.rl.skyrl_rewards import create_skyrl_reward_fn

        reward_fn = create_skyrl_reward_fn()
        assert callable(reward_fn)

    def test_reward_fn_math_correct(self):
        """Test math reward for correct answer."""
        from src.rl.skyrl_rewards import create_skyrl_reward_fn

        reward_fn = create_skyrl_reward_fn()

        responses = ["The answer is \\boxed{42}"]
        samples = [{"answer": "42", "task_type": "math"}]

        rewards = reward_fn(responses, samples)

        assert len(rewards) == 1
        assert rewards[0] == 1.0

    def test_reward_fn_math_incorrect(self):
        """Test math reward for incorrect answer."""
        from src.rl.skyrl_rewards import create_skyrl_reward_fn

        reward_fn = create_skyrl_reward_fn()

        responses = ["The answer is \\boxed{100}"]
        samples = [{"answer": "42", "task_type": "math"}]

        rewards = reward_fn(responses, samples)

        assert len(rewards) == 1
        assert rewards[0] == 0.0

    def test_reward_fn_logic_correct(self):
        """Test logic reward for correct answer with format."""
        from src.rl.skyrl_rewards import create_skyrl_reward_fn

        reward_fn = create_skyrl_reward_fn(require_think_tags=False)

        responses = ["<answer>yes</answer>"]
        samples = [{"answer": "yes", "task_type": "logic"}]

        rewards = reward_fn(responses, samples)

        assert len(rewards) == 1
        assert rewards[0] == 1.0

    def test_accuracy_reward_fn(self):
        """Test simple accuracy reward function."""
        from src.rl.skyrl_rewards import create_accuracy_reward_fn

        reward_fn = create_accuracy_reward_fn()

        responses = ["The answer is 42", "The answer is wrong"]
        samples = [
            {"answer": "42"},
            {"answer": "42"},
        ]

        rewards = reward_fn(responses, samples)

        assert rewards[0] == 1.0
        assert rewards[1] == 0.0

    def test_skyrl_reward_wrapper(self):
        """Test SkyRL reward wrapper with caching and stats."""
        from src.rl.skyrl_rewards import SkyRLRewardWrapper, create_accuracy_reward_fn

        wrapper = SkyRLRewardWrapper(
            base_fn=create_accuracy_reward_fn(),
            cache_enabled=True,
            track_stats=True,
        )

        responses = ["42", "wrong"]
        samples = [
            {"answer": "42", "sample_id": "1"},
            {"answer": "42", "sample_id": "2"},
        ]

        rewards = wrapper(responses, samples)

        assert rewards[0] == 1.0
        assert rewards[1] == 0.0
        assert wrapper.mean_reward == 0.5
        assert wrapper.accuracy == 0.5

        # Test cache (same response/sample should use cache)
        stats = wrapper.get_stats()
        assert stats["total_calls"] == 2


class TestSkyRLDataPipeline:
    """Tests for SkyRL data pipeline."""

    def test_convert_to_skyrl_format(self):
        """Test converting VerlSample to SkyRL format."""
        from src.rl.skyrl_data_pipeline import convert_to_skyrl_format
        from src.rl.verl_data_pipeline import VerlSample, VerlTaskType

        samples = [
            VerlSample(
                prompt="What is 2+2?",
                answer="4",
                task_type=VerlTaskType.MATH,
                sample_id="test_1",
                extra_info={"difficulty": "easy"},
            ),
        ]

        skyrl_data = convert_to_skyrl_format(samples)

        assert len(skyrl_data) == 1
        assert skyrl_data[0]["prompt"] == "What is 2+2?"
        assert skyrl_data[0]["metadata"]["answer"] == "4"
        assert skyrl_data[0]["metadata"]["task_type"] == "math"

    def test_samples_to_hf_dataset(self):
        """Test converting samples to HuggingFace Dataset."""
        from src.rl.skyrl_data_pipeline import samples_to_hf_dataset
        from src.rl.verl_data_pipeline import VerlSample, VerlTaskType

        samples = [
            VerlSample(
                prompt="What is 2+2?",
                answer="4",
                task_type=VerlTaskType.MATH,
                sample_id="test_1",
            ),
            VerlSample(
                prompt="Is the sky blue?",
                answer="yes",
                task_type=VerlTaskType.LOGIC,
                sample_id="test_2",
            ),
        ]

        dataset = samples_to_hf_dataset(samples)

        assert len(dataset) == 2
        assert "prompt" in dataset.column_names
        assert "answer" in dataset.column_names
        assert "task_type" in dataset.column_names

    def test_skyrl_prompt_dataset(self):
        """Test SkyRLPromptDataset wrapper."""
        from src.rl.skyrl_data_pipeline import SkyRLPromptDataset

        data = [
            {"prompt": "Q1", "metadata": {"answer": "A1"}},
            {"prompt": "Q2", "metadata": {"answer": "A2"}},
            {"prompt": "Q3", "metadata": {"answer": "A3"}},
        ]

        dataset = SkyRLPromptDataset(data)

        assert len(dataset) == 3
        assert dataset.get_prompt(0) == "Q1"
        assert dataset.get_metadata(1)["answer"] == "A2"

        # Test sampling
        sampled = dataset.sample(2, seed=42)
        assert len(sampled) == 2


class TestSkyRLTrainerUnit:
    """Unit tests for LORESkyRLTrainer (no GPU required)."""

    def test_trainer_initialization(self):
        """Test trainer initialization without loading model."""
        from src.rl.skyrl_trainer import SkyRLConfig, LORESkyRLTrainer

        config = SkyRLConfig(model_name="test-model")

        # Create trainer (lazy initialization - no model loaded yet)
        trainer = LORESkyRLTrainer(config=config)

        assert trainer.config.model_name == "test-model"
        assert trainer._model is None  # Lazy initialization
        assert trainer._tokenizer is None

    def test_trainer_with_custom_reward(self):
        """Test trainer with custom reward function."""
        from src.rl.skyrl_trainer import SkyRLConfig, LORESkyRLTrainer

        def custom_reward(responses, samples):
            return [0.5] * len(responses)

        trainer = LORESkyRLTrainer(
            config=SkyRLConfig(),
            reward_fn=custom_reward,
        )

        assert trainer._reward_fn == custom_reward
        assert trainer.reward_fn([" "], [{}]) == [0.5]


class TestSkyRLModuleExports:
    """Test that all exports are available."""

    def test_imports(self):
        """Test that all expected imports work."""
        from src.rl import (
            # Config
            SkyRLConfig,
            SkyRLTrainingResult,
            # Trainer
            LORESkyRLTrainer,
            train_skyrl_grpo,
            # Rewards
            create_skyrl_reward_fn,
            create_accuracy_reward_fn,
            create_format_aware_reward_fn,
            SkyRLRewardWrapper,
            # Data
            convert_to_skyrl_format,
            load_skyrl_datasets,
            load_smoke_test_datasets,
            SkyRLPromptDataset,
        )

        # Verify they're callable/classes
        assert SkyRLConfig is not None
        assert LORESkyRLTrainer is not None
        assert callable(create_skyrl_reward_fn)
        assert callable(load_skyrl_datasets)


# Skip integration tests that require GPU/heavy dependencies
@pytest.mark.skip(reason="Requires GPU and heavy dependencies")
class TestSkyRLIntegration:
    """Integration tests for full SkyRL training pipeline."""

    def test_smoke_training(self):
        """Test minimal training run."""
        pass

    def test_vllm_generation(self):
        """Test vLLM batched generation."""
        pass
