"""Data loading and gym building for LORE experiments.

This module provides:
- Dataset loaders for GSM8K, HumanEval, and BBH
- Hidden goal injection for Regime B training
- Stratified train/val/test splitting
- Versioned JSONL export
"""

from src.data.loaders import (
    DataLoader,
    DataPoint,
    DatasetConfig,
    get_eval_dataset,
    load_dataset,
)
from src.data.gym_builder import (
    BBHLoader,
    DatasetSplit,
    GoalCategory,
    GSM8KLoader,
    GymBuilder,
    GymConfig,
    GymSample,
    GymTask,
    HiddenGoal,
    HiddenGoalInjector,
    HiddenGoalSpec,
    HumanEvalLoader,
    StratifiedSplitter,
    TaskType,
    build_and_save_gym,
)

__all__ = [
    # Loaders
    "DataLoader",
    "DataPoint",
    "DatasetConfig",
    "load_dataset",
    "get_eval_dataset",
    # Gym builder
    "GymBuilder",
    "GymConfig",
    "GymSample",
    "GymTask",
    # Dataset-specific loaders
    "GSM8KLoader",
    "HumanEvalLoader",
    "BBHLoader",
    # Hidden goals
    "HiddenGoal",
    "HiddenGoalSpec",
    "HiddenGoalInjector",
    "GoalCategory",
    # Types
    "TaskType",
    "DatasetSplit",
    # Utilities
    "StratifiedSplitter",
    "build_and_save_gym",
]
