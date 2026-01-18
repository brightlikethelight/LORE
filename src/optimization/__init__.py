"""Prompt optimization methods for LORE experiments.

This module provides:
- GEPA: Evolutionary prompt optimization using the official GEPA library
- PromptBreeder: Genetic algorithm-based prompt optimization
- MIPRO: Multi-objective prompt optimization

Usage:
    from src.optimization import GEPA, EvaluationSample, TaskType

    optimizer = GEPA(
        task_lm="openai/gpt-4o-mini",
        reflection_lm="openai/gpt-4o",
        regime="A",
    )

    result = await optimizer.optimize(trainset, valset)
"""

from src.optimization.base_optimizer import BaseOptimizer, OptimizationResult, PromptIndividual
from src.optimization.gepa import (
    # Core classes
    GEPA,
    GRG,
    GEPAResult,
    LOREOptimizationResult,
    LOREGEPAAdapter,
    # Enums and types
    Regime,
    TaskType,
    EvaluationSample,
    FitnessComponents,
    # Components
    MultiTaskEvaluator,
    StealthMonitor,
    MutationOperator,
    # Functions
    run_gepa_optimization,
    run_lore_optimization,
    create_dspy_gepa_optimizer,
    create_lore_metric,
    get_optimizer,
    print_installation_guide,
)
from src.optimization.mipro import MIPROOptimizer
from src.optimization.promptbreeder import PromptBreederOptimizer
from src.optimization.ultrainteract_gepa import (
    # Types
    UIFitnessComponents,
    UIOptimizationResult,
    # Adapter
    UltraInteractGEPAAdapter,
    # Constants
    TASK_TEMPLATES,
    DEFAULT_SEED_PROMPT,
    # Functions
    run_ultrainteract_evolution,
    get_composed_prompt,
    load_evolved_prompt,
)
from src.optimization.grg_mutations import (
    # Constants
    ATOMIC_MODULES,
    ATOMIC_MODULES_BY_CATEGORY,
    # Types
    FitnessHistory,
    FitnessRecord,
    MutationLog,
    MutationResult,
    MutationType,
    # Base class
    GRGMutation,
    # Mutation implementations
    DirectMutation,
    EDAMutation,
    HypermutationReinit,
    LamarckianMutation,
    PromptCrossover,
    # Registry and orchestration
    MUTATION_REGISTRY,
    MutationOrchestrator,
    create_mutation,
    # Utilities
    sample_seed_grg,
    get_atomic_modules_by_category,
    list_all_categories,
)

__all__ = [
    # Base classes
    "BaseOptimizer",
    "OptimizationResult",
    "PromptIndividual",
    # GEPA core
    "GEPA",
    "GRG",
    "GEPAResult",
    "LOREOptimizationResult",
    "LOREGEPAAdapter",
    # Enums and types
    "Regime",
    "TaskType",
    "EvaluationSample",
    "FitnessComponents",
    # Components
    "MultiTaskEvaluator",
    "StealthMonitor",
    "MutationOperator",
    # Functions
    "run_gepa_optimization",
    "run_lore_optimization",
    "create_dspy_gepa_optimizer",
    "create_lore_metric",
    "get_optimizer",
    "print_installation_guide",
    # Other optimizers
    "PromptBreederOptimizer",
    "MIPROOptimizer",
    # UltraInteract GEPA
    "UIFitnessComponents",
    "UIOptimizationResult",
    "UltraInteractGEPAAdapter",
    "TASK_TEMPLATES",
    "DEFAULT_SEED_PROMPT",
    "run_ultrainteract_evolution",
    "get_composed_prompt",
    "load_evolved_prompt",
    # GRG Mutations - Constants
    "ATOMIC_MODULES",
    "ATOMIC_MODULES_BY_CATEGORY",
    # GRG Mutations - Types
    "FitnessHistory",
    "FitnessRecord",
    "MutationLog",
    "MutationResult",
    "MutationType",
    # GRG Mutations - Base class
    "GRGMutation",
    # GRG Mutations - Implementations
    "DirectMutation",
    "EDAMutation",
    "HypermutationReinit",
    "LamarckianMutation",
    "PromptCrossover",
    # GRG Mutations - Registry
    "MUTATION_REGISTRY",
    "MutationOrchestrator",
    "create_mutation",
    # GRG Mutations - Utilities
    "sample_seed_grg",
    "get_atomic_modules_by_category",
    "list_all_categories",
]
