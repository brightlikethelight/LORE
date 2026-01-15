"""inspect_ai task definitions for LORE evaluation."""

from evals.tasks.math_reasoning import math_reasoning_task
from evals.tasks.science_reasoning import science_reasoning_task
from evals.tasks.competition_math import competition_math_task

__all__ = [
    "math_reasoning_task",
    "science_reasoning_task",
    "competition_math_task",
]
