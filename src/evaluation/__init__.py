"""Evaluation modules for LORE experiments.

This module provides:
- CRH integration for monitorability evaluation
- Metrics for comparing baseline vs evolved prompts
- Math grading with sympy-based equivalence checking
"""

from src.evaluation.crh_evaluator import (
    CRHEvaluator,
    CRHMetrics,
    CRHComparisonResult,
    MonitorabilityDelta,
    CRHSideTask,
    run_crh_evaluation,
    compare_prompt_monitorability,
)
from src.evaluation.math_grader import (
    math_equal,
    grade_math_answer,
    extract_boxed_answer,
    extract_math_answer,
)

__all__ = [
    # CRH evaluation
    "CRHEvaluator",
    "CRHMetrics",
    "CRHComparisonResult",
    "MonitorabilityDelta",
    "CRHSideTask",
    "run_crh_evaluation",
    "compare_prompt_monitorability",
    # Math grading
    "math_equal",
    "grade_math_answer",
    "extract_boxed_answer",
    "extract_math_answer",
]
