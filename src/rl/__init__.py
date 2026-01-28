"""Reinforcement Learning module for LORE.

This module provides GRPO (Group Relative Policy Optimization) training
using three frameworks:
1. TRL (legacy) - UltraInteract dataset
2. Verl - OpenR1-Math-220k and SynLogic datasets (requires Ray cluster)
3. SkyRL (recommended for single-GPU) - Faster async batched generation

Key components:

TRL-based (legacy):
- config: Configuration dataclasses for GRPO and LoRA
- reward: Task-based reward functions for math, coding, and logic
- data_pipeline: Dataset adapter for TRL GRPO format
- grpo_trainer: GRPO training wrapper with LoRA support

Verl-based (multi-GPU with Ray):
- verl_trainer: Verl GRPO trainer for OpenR1-Math and SynLogic
- verl_rewards: Reward functions using math-verify and rule-based verifiers
- verl_data_pipeline: Dataset loaders for OpenR1-Math-220k and SynLogic

SkyRL-based (recommended for single-GPU):
- skyrl_trainer: SkyRL GRPO trainer with async batched generation
- skyrl_rewards: Reward function wrappers for SkyRL interface
- skyrl_data_pipeline: Dataset adapters for SkyRL format

- evaluation: Evaluation pipeline for RL-trained models
"""

# TRL-based (legacy)
from src.rl.config import GRPOConfig, LoRAConfig, RLConfig, load_config
from src.rl.data_pipeline import (
    create_grpo_dataset,
    format_prompt,
    UltraInteractGRPODataset,
)
from src.rl.grpo_trainer import LOREGRPOTrainer
from src.rl.reward import (
    compute_reward,
    RewardFunction,
    TaskRewardFunction,
)
from src.rl.evaluation import (
    evaluate_model,
    compare_models,
    RLEvaluationResult,
)

# Verl-based (recommended)
from src.rl.verl_trainer import (
    VerlGRPOConfig,
    VerlGRPOTrainer,
    VerlTrainingResult,
    train_verl_grpo,
)
from src.rl.verl_rewards import (
    MathVerifyReward,
    SynLogicReward,
    CodeExecutionReward,
    MultiTaskReward,
    create_verl_reward_function,
)
from src.rl.verl_data_pipeline import (
    VerlSample,
    VerlTaskType,
    load_openr1_math,
    load_synlogic,
    load_code_contests,
    load_verl_datasets,
)

# SkyRL-based (recommended for single-GPU)
from src.rl.skyrl_trainer import (
    SkyRLConfig,
    SkyRLTrainingResult,
    LORESkyRLTrainer,
    train_skyrl_grpo,
)
from src.rl.skyrl_rewards import (
    create_skyrl_reward_fn,
    create_accuracy_reward_fn,
    create_format_aware_reward_fn,
    SkyRLRewardWrapper,
)
from src.rl.skyrl_data_pipeline import (
    convert_to_skyrl_format,
    load_skyrl_datasets,
    load_smoke_test_datasets,
    SkyRLPromptDataset,
)

__all__ = [
    # TRL Config (legacy)
    "GRPOConfig",
    "LoRAConfig",
    "RLConfig",
    "load_config",
    # TRL Data (legacy)
    "create_grpo_dataset",
    "format_prompt",
    "UltraInteractGRPODataset",
    # TRL Training (legacy)
    "LOREGRPOTrainer",
    # TRL Reward (legacy)
    "compute_reward",
    "RewardFunction",
    "TaskRewardFunction",
    # Verl Config
    "VerlGRPOConfig",
    # Verl Data
    "VerlSample",
    "VerlTaskType",
    "load_openr1_math",
    "load_synlogic",
    "load_code_contests",
    "load_verl_datasets",
    # Verl Training
    "VerlGRPOTrainer",
    "VerlTrainingResult",
    "train_verl_grpo",
    # Verl Rewards
    "MathVerifyReward",
    "SynLogicReward",
    "CodeExecutionReward",
    "MultiTaskReward",
    "create_verl_reward_function",
    # Evaluation
    "evaluate_model",
    "compare_models",
    "RLEvaluationResult",
    # SkyRL Config
    "SkyRLConfig",
    "SkyRLTrainingResult",
    # SkyRL Trainer
    "LORESkyRLTrainer",
    "train_skyrl_grpo",
    # SkyRL Rewards
    "create_skyrl_reward_fn",
    "create_accuracy_reward_fn",
    "create_format_aware_reward_fn",
    "SkyRLRewardWrapper",
    # SkyRL Data
    "convert_to_skyrl_format",
    "load_skyrl_datasets",
    "load_smoke_test_datasets",
    "SkyRLPromptDataset",
]
