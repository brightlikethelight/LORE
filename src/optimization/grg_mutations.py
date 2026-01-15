"""GRG (Generalized Reasoning Guidelines) Mutation Operators.

This module implements the 5 PromptBreeder mutation classes adapted for GRG evolution:

1. DirectMutation: Ask optimizer to directly improve the GRG
2. EDAMutation: Estimate distribution of high-performing GRGs, sample new ones
3. HypermutationReinit: Occasionally reset mutation strategy itself
4. LamarckianMutation: Use task feedback to guide specific improvements
5. PromptCrossover: Combine principles from two high-fitness GRGs

Each mutation:
- Takes a GRG string and fitness history
- Returns a new GRG string
- Logs the mutation type and rationale for analysis

Also includes Self-Discover's reasoning module vocabulary as seed principles.

Reference:
- Fernando et al., "PromptBreeder: Self-Referential Self-Improvement" (2023)
- Zhou et al., "Self-Discover: Large Language Models Self-Compose Reasoning" (2024)
"""

from __future__ import annotations

import json
import logging
import random
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, TypedDict

# =============================================================================
# Self-Discover Atomic Modules (39 Reasoning Principles)
# =============================================================================

# These are the 39 atomic reasoning modules from the Self-Discover paper
# (Zhou et al., 2024) that serve as seed principles for GRG evolution.
ATOMIC_MODULES: list[str] = [
    # Problem Decomposition (1-5)
    "Break the problem into smaller subproblems",
    "Identify the simplest case first, then generalize",
    "Work backwards from the desired result",
    "Divide the problem into independent parts that can be solved separately",
    "Identify which parts of the problem are already solved or given",
    # Constraint Handling (6-10)
    "Identify the key constraints before solving",
    "List all constraints explicitly and check each one",
    "Determine which constraints are most restrictive",
    "Look for hidden assumptions or implicit constraints",
    "Consider what happens at the boundary conditions",
    # Edge Cases & Verification (11-15)
    "Consider edge cases explicitly",
    "Test your solution with extreme values",
    "Check if the solution works for the simplest possible input",
    "Verify each step before proceeding",
    "Ask yourself: could this step possibly be wrong?",
    # Examples & Concretization (16-20)
    "Use concrete examples to check reasoning",
    "Create a specific example and trace through it",
    "Draw a diagram or visualization if helpful",
    "Substitute actual numbers to verify abstract reasoning",
    "Generate counterexamples to test your hypothesis",
    # Abstraction & Patterns (21-25)
    "Look for patterns in the problem structure",
    "Abstract away unnecessary details",
    "Identify what type of problem this is",
    "Recall similar problems you have solved before",
    "Map the problem to a known framework or technique",
    # Logical Reasoning (26-30)
    "Make all implicit assumptions explicit",
    "Consider the contrapositive of each statement",
    "Identify necessary vs. sufficient conditions",
    "Check if your conclusion follows logically from premises",
    "Look for logical dependencies between statements",
    # Exploration & Search (31-35)
    "Consider multiple approaches before committing to one",
    "Explore the solution space systematically",
    "Use elimination to narrow down possibilities",
    "Consider what information would help most",
    "Ask: what is the key insight needed here?",
    # Self-Monitoring & Meta-cognition (36-39)
    "Pause and reflect: is my current approach working?",
    "Identify where you are stuck and why",
    "Summarize what you know so far",
    "Check if you have answered the actual question asked",
]

# Grouped by category for more structured sampling
ATOMIC_MODULES_BY_CATEGORY: dict[str, list[str]] = {
    "problem_decomposition": ATOMIC_MODULES[0:5],
    "constraint_handling": ATOMIC_MODULES[5:10],
    "edge_cases_verification": ATOMIC_MODULES[10:15],
    "examples_concretization": ATOMIC_MODULES[15:20],
    "abstraction_patterns": ATOMIC_MODULES[20:25],
    "logical_reasoning": ATOMIC_MODULES[25:30],
    "exploration_search": ATOMIC_MODULES[30:35],
    "self_monitoring": ATOMIC_MODULES[35:39],
}


# =============================================================================
# Type Definitions
# =============================================================================


class FitnessRecord(TypedDict):
    """A single fitness evaluation record."""

    fitness: float
    generation: int
    task_type: str | None
    task_id: str | None
    error_type: str | None
    feedback: str | None


@dataclass
class FitnessHistory:
    """Fitness history for a GRG with detailed records."""

    records: list[FitnessRecord] = field(default_factory=list)
    best_fitness: float = 0.0
    worst_fitness: float = 1.0
    mean_fitness: float = 0.0
    fitness_trend: str = "unknown"  # "improving", "degrading", "stable"

    def add_record(self, record: FitnessRecord) -> None:
        """Add a fitness record and update statistics."""
        self.records.append(record)
        fitnesses = [r["fitness"] for r in self.records]
        self.best_fitness = max(fitnesses)
        self.worst_fitness = min(fitnesses)
        self.mean_fitness = sum(fitnesses) / len(fitnesses)

        # Calculate trend from recent history
        if len(self.records) >= 5:
            recent = [r["fitness"] for r in self.records[-5:]]
            older = [r["fitness"] for r in self.records[-10:-5]] if len(self.records) >= 10 else []
            if older:
                recent_mean = sum(recent) / len(recent)
                older_mean = sum(older) / len(older)
                if recent_mean > older_mean + 0.01:
                    self.fitness_trend = "improving"
                elif recent_mean < older_mean - 0.01:
                    self.fitness_trend = "degrading"
                else:
                    self.fitness_trend = "stable"

    def get_error_patterns(self) -> dict[str, int]:
        """Get frequency of error types from history."""
        patterns: dict[str, int] = {}
        for record in self.records:
            if record.get("error_type"):
                error = record["error_type"]
                patterns[error] = patterns.get(error, 0) + 1
        return patterns

    def get_task_performance(self) -> dict[str, float]:
        """Get mean fitness by task type."""
        by_task: dict[str, list[float]] = {}
        for record in self.records:
            task = record.get("task_type", "unknown")
            if task:
                if task not in by_task:
                    by_task[task] = []
                by_task[task].append(record["fitness"])

        return {task: sum(fits) / len(fits) for task, fits in by_task.items()}


@dataclass
class MutationLog:
    """Log entry for a mutation operation."""

    mutation_id: str
    mutation_type: str
    timestamp: str
    parent_grg: str
    child_grg: str
    rationale: str
    fitness_before: float | None = None
    fitness_after: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "mutation_id": self.mutation_id,
            "mutation_type": self.mutation_type,
            "timestamp": self.timestamp,
            "parent_grg": self.parent_grg,
            "child_grg": self.child_grg,
            "rationale": self.rationale,
            "fitness_before": self.fitness_before,
            "fitness_after": self.fitness_after,
            "metadata": self.metadata,
        }


class MutationResult(TypedDict):
    """Result of a mutation operation."""

    grg: str
    mutation_type: str
    rationale: str
    mutation_id: str
    parent_id: str | None


# =============================================================================
# LLM Client Protocol
# =============================================================================


class LLMClient(Protocol):
    """Protocol for LLM completion clients."""

    async def complete(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a completion for the given prompt."""
        ...


# =============================================================================
# Base Mutation Class
# =============================================================================


class GRGMutation(ABC):
    """Abstract base class for GRG mutation operators.

    All mutation operators inherit from this class and implement
    the `mutate` method to produce variations of GRGs.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        seed: int = 42,
        log_dir: Path | None = None,
    ) -> None:
        """Initialize the mutation operator.

        Args:
            llm_client: LLM client for generating mutations
            seed: Random seed for reproducibility
            log_dir: Directory to save mutation logs
        """
        self.llm = llm_client
        self.seed = seed
        self.rng = random.Random(seed)
        self.log_dir = log_dir

        self.logger = logging.getLogger(self.__class__.__name__)
        self.mutation_logs: list[MutationLog] = []

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    @property
    @abstractmethod
    def mutation_type(self) -> str:
        """Return the type identifier for this mutation."""
        ...

    @abstractmethod
    async def mutate(
        self,
        grg: str,
        fitness_history: FitnessHistory,
        **kwargs: Any,
    ) -> MutationResult:
        """Apply mutation to produce a new GRG.

        Args:
            grg: The parent GRG string to mutate
            fitness_history: History of fitness evaluations
            **kwargs: Additional mutation-specific parameters

        Returns:
            MutationResult containing the new GRG and metadata
        """
        ...

    def _generate_mutation_id(self) -> str:
        """Generate a unique mutation ID."""
        return f"{self.mutation_type}_{uuid.uuid4().hex[:8]}"

    def _log_mutation(
        self,
        parent_grg: str,
        child_grg: str,
        rationale: str,
        fitness_before: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MutationLog:
        """Log a mutation for analysis.

        Args:
            parent_grg: The original GRG
            child_grg: The mutated GRG
            rationale: Explanation of why this mutation was made
            fitness_before: Fitness of parent GRG
            metadata: Additional metadata

        Returns:
            MutationLog entry
        """
        log = MutationLog(
            mutation_id=self._generate_mutation_id(),
            mutation_type=self.mutation_type,
            timestamp=datetime.now().isoformat(),
            parent_grg=parent_grg,
            child_grg=child_grg,
            rationale=rationale,
            fitness_before=fitness_before,
            metadata=metadata or {},
        )

        self.mutation_logs.append(log)

        self.logger.info(
            f"Mutation [{self.mutation_type}]: {rationale[:100]}..."
            if len(rationale) > 100
            else f"Mutation [{self.mutation_type}]: {rationale}"
        )

        # Save to file if log_dir is set
        if self.log_dir:
            log_file = self.log_dir / f"mutations_{datetime.now().strftime('%Y%m%d')}.jsonl"
            with open(log_file, "a") as f:
                f.write(json.dumps(log.to_dict()) + "\n")

        return log

    def save_logs(self, path: Path) -> None:
        """Save all mutation logs to a file."""
        with open(path, "w") as f:
            for log in self.mutation_logs:
                f.write(json.dumps(log.to_dict()) + "\n")


# =============================================================================
# Mutation Implementations
# =============================================================================


class DirectMutation(GRGMutation):
    """Direct mutation: Ask optimizer to directly improve the GRG.

    This is the simplest mutation operator. It presents the current GRG
    to the LLM and asks it to generate an improved version based on
    general quality criteria.
    """

    @property
    def mutation_type(self) -> str:
        return "direct"

    async def mutate(
        self,
        grg: str,
        fitness_history: FitnessHistory,
        improvement_focus: str | None = None,
        **kwargs: Any,
    ) -> MutationResult:
        """Directly ask LLM to improve the GRG.

        Args:
            grg: Current GRG to improve
            fitness_history: Fitness history for context
            improvement_focus: Optional specific aspect to focus on

        Returns:
            MutationResult with improved GRG
        """
        focus = improvement_focus or self._select_focus(fitness_history)

        prompt = f"""You are an expert at crafting reasoning guidelines for AI systems.

Below is a Generalized Reasoning Guideline (GRG) that instructs an AI how to approach problems:

<current_grg>
{grg}
</current_grg>

Current fitness statistics:
- Mean fitness: {fitness_history.mean_fitness:.3f}
- Best fitness: {fitness_history.best_fitness:.3f}
- Trend: {fitness_history.fitness_trend}

Your task: Generate an IMPROVED version of this GRG that is more effective at guiding reasoning.

Focus on: {focus}

Requirements:
1. Preserve the core intent while improving clarity and effectiveness
2. Make instructions more actionable and specific
3. Remove redundancy or vague language
4. The new GRG should be similar length (within 20%)

Output ONLY the improved GRG, no explanations:"""

        new_grg = await self.llm.complete(prompt, max_tokens=1000, temperature=0.7)
        new_grg = new_grg.strip()

        rationale = f"Direct improvement focused on: {focus}"

        log = self._log_mutation(
            parent_grg=grg,
            child_grg=new_grg,
            rationale=rationale,
            fitness_before=fitness_history.mean_fitness,
            metadata={"focus": focus, "trend": fitness_history.fitness_trend},
        )

        return {
            "grg": new_grg,
            "mutation_type": self.mutation_type,
            "rationale": rationale,
            "mutation_id": log.mutation_id,
            "parent_id": None,
        }

    def _select_focus(self, history: FitnessHistory) -> str:
        """Select improvement focus based on fitness history."""
        focuses = [
            "clarity and precision of instructions",
            "handling of edge cases and exceptions",
            "step-by-step structure and organization",
            "verification and self-checking guidance",
            "problem decomposition strategies",
        ]

        # Bias toward certain focuses based on performance
        if history.fitness_trend == "degrading":
            focuses.insert(0, "fundamental reasoning soundness")
        elif history.fitness_trend == "stable":
            focuses.insert(0, "novel reasoning strategies")

        return self.rng.choice(focuses)


class EDAMutation(GRGMutation):
    """EDA (Estimation of Distribution Algorithm) mutation.

    This operator analyzes high-performing GRGs to estimate what makes
    them effective, then samples new GRGs from this learned distribution.
    """

    @property
    def mutation_type(self) -> str:
        return "eda"

    async def mutate(
        self,
        grg: str,
        fitness_history: FitnessHistory,
        elite_grgs: list[tuple[str, float]] | None = None,
        **kwargs: Any,
    ) -> MutationResult:
        """Generate new GRG based on distribution of high performers.

        Args:
            grg: Current GRG (used as baseline)
            fitness_history: Fitness history
            elite_grgs: List of (grg_string, fitness) tuples for top performers

        Returns:
            MutationResult with sampled GRG
        """
        # Use provided elites or seed modules
        if elite_grgs and len(elite_grgs) >= 3:
            elite_examples = "\n".join(
                [f"[Fitness: {fit:.3f}]\n{g}" for g, fit in elite_grgs[:5]]
            )
            source = "elite_population"
        else:
            # Fall back to sampling from atomic modules
            sampled = self.rng.sample(ATOMIC_MODULES, min(8, len(ATOMIC_MODULES)))
            elite_examples = "\n".join([f"- {m}" for m in sampled])
            source = "atomic_modules"

        prompt = f"""You are analyzing patterns in effective reasoning guidelines.

Here are examples of high-performing Generalized Reasoning Guidelines (GRGs):

{elite_examples}

Your task: Generate a NEW GRG that captures the best qualities of these examples.

Analysis steps:
1. Identify common patterns across the high performers
2. Note what makes them effective (structure, specificity, coverage)
3. Synthesize a novel GRG that combines these strengths

Requirements:
- Must be distinct from all examples (not a copy)
- Should capture the "essence" of what makes them work
- Be specific and actionable, not generic

Output ONLY the new GRG:"""

        new_grg = await self.llm.complete(prompt, max_tokens=1000, temperature=0.9)
        new_grg = new_grg.strip()

        rationale = f"EDA-sampled from {source}, combining patterns from {len(elite_examples.splitlines())} examples"

        log = self._log_mutation(
            parent_grg=grg,
            child_grg=new_grg,
            rationale=rationale,
            fitness_before=fitness_history.mean_fitness,
            metadata={
                "source": source,
                "num_examples": len(elite_grgs) if elite_grgs else 0,
            },
        )

        return {
            "grg": new_grg,
            "mutation_type": self.mutation_type,
            "rationale": rationale,
            "mutation_id": log.mutation_id,
            "parent_id": None,
        }


class HypermutationReinit(GRGMutation):
    """Hypermutation with occasional reinitialization.

    This operator occasionally resets the mutation strategy itself,
    introducing more radical changes to escape local optima.
    It can either:
    1. Generate a completely new GRG from scratch
    2. Hybridize with random atomic modules
    3. Apply extreme transformation to existing GRG
    """

    @property
    def mutation_type(self) -> str:
        return "hypermutation_reinit"

    def __init__(
        self,
        llm_client: LLMClient,
        seed: int = 42,
        log_dir: Path | None = None,
        reinit_probability: float = 0.3,
    ) -> None:
        super().__init__(llm_client, seed, log_dir)
        self.reinit_probability = reinit_probability
        self.mutation_strategy: str = "transform"  # Current strategy
        self.strategy_history: list[str] = []

    async def mutate(
        self,
        grg: str,
        fitness_history: FitnessHistory,
        force_reinit: bool = False,
        **kwargs: Any,
    ) -> MutationResult:
        """Apply hypermutation, potentially reinitializing strategy.

        Args:
            grg: Current GRG
            fitness_history: Fitness history
            force_reinit: Force a reinitialization regardless of probability

        Returns:
            MutationResult with mutated/reinitialized GRG
        """
        # Decide whether to reinitialize
        should_reinit = force_reinit or (self.rng.random() < self.reinit_probability)

        if should_reinit:
            # Reset mutation strategy
            strategies = ["from_scratch", "hybridize", "radical_transform"]
            old_strategy = self.mutation_strategy
            self.mutation_strategy = self.rng.choice(strategies)
            self.strategy_history.append(self.mutation_strategy)

            self.logger.info(
                f"Hypermutation reinit: {old_strategy} -> {self.mutation_strategy}"
            )

        # Apply the current strategy
        if self.mutation_strategy == "from_scratch":
            new_grg, rationale = await self._from_scratch()
        elif self.mutation_strategy == "hybridize":
            new_grg, rationale = await self._hybridize(grg)
        else:
            new_grg, rationale = await self._radical_transform(grg, fitness_history)

        log = self._log_mutation(
            parent_grg=grg,
            child_grg=new_grg,
            rationale=rationale,
            fitness_before=fitness_history.mean_fitness,
            metadata={
                "strategy": self.mutation_strategy,
                "was_reinit": should_reinit,
                "strategy_history_len": len(self.strategy_history),
            },
        )

        return {
            "grg": new_grg,
            "mutation_type": self.mutation_type,
            "rationale": rationale,
            "mutation_id": log.mutation_id,
            "parent_id": None,
        }

    async def _from_scratch(self) -> tuple[str, str]:
        """Generate a completely new GRG from scratch."""
        # Sample diverse seed modules
        categories = list(ATOMIC_MODULES_BY_CATEGORY.keys())
        self.rng.shuffle(categories)
        seeds = []
        for cat in categories[:4]:
            seeds.append(self.rng.choice(ATOMIC_MODULES_BY_CATEGORY[cat]))

        prompt = f"""You are creating a novel reasoning guideline from first principles.

Seed principles to incorporate:
{chr(10).join(f'- {s}' for s in seeds)}

Generate a comprehensive Generalized Reasoning Guideline (GRG) that:
1. Synthesizes these principles into a cohesive framework
2. Adds your own novel reasoning strategies
3. Is structured for clear, step-by-step application
4. Covers problem analysis, solution development, and verification

Output ONLY the GRG:"""

        new_grg = await self.llm.complete(prompt, max_tokens=1200, temperature=0.95)
        rationale = f"Generated from scratch using seeds from: {categories[:4]}"

        return new_grg.strip(), rationale

    async def _hybridize(self, grg: str) -> tuple[str, str]:
        """Hybridize existing GRG with random atomic modules."""
        # Select random modules from different categories
        modules_to_inject = []
        for category in self.rng.sample(list(ATOMIC_MODULES_BY_CATEGORY.keys()), 3):
            modules_to_inject.append(
                self.rng.choice(ATOMIC_MODULES_BY_CATEGORY[category])
            )

        prompt = f"""You are enhancing a reasoning guideline with new principles.

Current GRG:
{grg}

New principles to integrate:
{chr(10).join(f'- {m}' for m in modules_to_inject)}

Create a HYBRID GRG that:
1. Preserves the strongest elements of the original
2. Seamlessly integrates the new principles
3. Removes any redundancy
4. Maintains coherent structure

Output ONLY the hybrid GRG:"""

        new_grg = await self.llm.complete(prompt, max_tokens=1200, temperature=0.85)
        rationale = f"Hybridized with modules: {modules_to_inject}"

        return new_grg.strip(), rationale

    async def _radical_transform(
        self, grg: str, history: FitnessHistory
    ) -> tuple[str, str]:
        """Apply radical transformation to escape local optima."""
        transform_types = [
            "invert the order of reasoning steps",
            "focus exclusively on verification and error-checking",
            "emphasize exploration over exploitation",
            "adopt a more formal, mathematical structure",
            "shift to a more intuitive, example-driven approach",
        ]

        transform = self.rng.choice(transform_types)

        prompt = f"""You are radically transforming a reasoning guideline.

Current GRG:
{grg}

Current performance is stagnating (trend: {history.fitness_trend}).

Transformation directive: {transform}

Generate a RADICALLY DIFFERENT GRG that:
1. Applies the transformation directive
2. Breaks away from the current approach
3. Explores a fundamentally different reasoning style
4. Still aims to be effective for problem-solving

Output ONLY the transformed GRG:"""

        new_grg = await self.llm.complete(prompt, max_tokens=1200, temperature=0.95)
        rationale = f"Radical transform: {transform}"

        return new_grg.strip(), rationale


class LamarckianMutation(GRGMutation):
    """Lamarckian mutation: Use task feedback to guide specific improvements.

    This mutation operator analyzes specific task failures and feedback
    to make targeted improvements to the GRG. Named after Lamarckism
    where acquired characteristics (learned from experience) are inherited.
    """

    @property
    def mutation_type(self) -> str:
        return "lamarckian"

    async def mutate(
        self,
        grg: str,
        fitness_history: FitnessHistory,
        recent_feedback: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> MutationResult:
        """Mutate GRG based on task-specific feedback.

        Args:
            grg: Current GRG
            fitness_history: Fitness history with error patterns
            recent_feedback: List of recent task feedback dicts with
                             keys: task_id, task_type, success, error_type, feedback

        Returns:
            MutationResult with feedback-guided improvements
        """
        # Analyze error patterns
        error_patterns = fitness_history.get_error_patterns()
        task_performance = fitness_history.get_task_performance()

        # Build feedback summary
        feedback_summary = self._summarize_feedback(
            error_patterns, task_performance, recent_feedback
        )

        prompt = f"""You are improving a reasoning guideline based on specific performance feedback.

Current GRG:
{grg}

Performance Analysis:
{feedback_summary}

Your task: Generate an IMPROVED GRG that specifically addresses the identified weaknesses.

Strategy:
1. Identify which parts of the GRG are causing the failures
2. Add specific guidance to handle the problematic cases
3. Strengthen weak areas while preserving strengths
4. Add explicit error-prevention mechanisms where needed

Output ONLY the improved GRG:"""

        new_grg = await self.llm.complete(prompt, max_tokens=1200, temperature=0.7)
        new_grg = new_grg.strip()

        # Generate detailed rationale
        improvements = []
        if error_patterns:
            top_errors = sorted(error_patterns.items(), key=lambda x: -x[1])[:3]
            improvements.extend([f"address {err} errors" for err, _ in top_errors])

        weak_tasks = [t for t, p in task_performance.items() if p < 0.5]
        if weak_tasks:
            improvements.append(f"improve on {', '.join(weak_tasks[:2])}")

        rationale = f"Lamarckian adaptation: {'; '.join(improvements) if improvements else 'general improvement'}"

        log = self._log_mutation(
            parent_grg=grg,
            child_grg=new_grg,
            rationale=rationale,
            fitness_before=fitness_history.mean_fitness,
            metadata={
                "error_patterns": error_patterns,
                "weak_tasks": weak_tasks,
                "num_feedback_records": len(recent_feedback) if recent_feedback else 0,
            },
        )

        return {
            "grg": new_grg,
            "mutation_type": self.mutation_type,
            "rationale": rationale,
            "mutation_id": log.mutation_id,
            "parent_id": None,
        }

    def _summarize_feedback(
        self,
        error_patterns: dict[str, int],
        task_performance: dict[str, float],
        recent_feedback: list[dict[str, Any]] | None,
    ) -> str:
        """Create a summary of performance feedback for the prompt."""
        lines = []

        # Error patterns
        if error_patterns:
            lines.append("Common Error Types:")
            for error, count in sorted(error_patterns.items(), key=lambda x: -x[1])[:5]:
                lines.append(f"  - {error}: {count} occurrences")

        # Task-specific performance
        if task_performance:
            lines.append("\nPerformance by Task Type:")
            for task, perf in sorted(task_performance.items(), key=lambda x: x[1]):
                status = "WEAK" if perf < 0.5 else "OK" if perf < 0.8 else "STRONG"
                lines.append(f"  - {task}: {perf:.2f} ({status})")

        # Recent specific feedback
        if recent_feedback:
            lines.append("\nRecent Specific Feedback:")
            for fb in recent_feedback[-5:]:
                if fb.get("feedback"):
                    task_type = fb.get("task_type", "unknown")
                    success = "✓" if fb.get("success") else "✗"
                    lines.append(f"  [{success}] {task_type}: {fb['feedback'][:100]}")

        return "\n".join(lines) if lines else "No specific feedback available."


class PromptCrossover(GRGMutation):
    """Crossover mutation: Combine principles from two high-fitness GRGs.

    This operator performs "sexual reproduction" of GRGs by combining
    effective elements from two parent GRGs to produce offspring.
    """

    @property
    def mutation_type(self) -> str:
        return "crossover"

    async def mutate(
        self,
        grg: str,
        fitness_history: FitnessHistory,
        second_parent: str | None = None,
        second_parent_fitness: float | None = None,
        **kwargs: Any,
    ) -> MutationResult:
        """Combine two GRGs through crossover.

        Args:
            grg: First parent GRG
            fitness_history: Fitness history for first parent
            second_parent: Second parent GRG (required)
            second_parent_fitness: Fitness of second parent

        Returns:
            MutationResult with offspring GRG
        """
        if not second_parent:
            # Fall back to hybridizing with atomic modules
            second_parent = "\n".join(
                self.rng.sample(ATOMIC_MODULES, min(5, len(ATOMIC_MODULES)))
            )
            second_parent_fitness = 0.5

        # Determine crossover strategy based on fitness comparison
        parent1_fitness = fitness_history.mean_fitness
        parent2_fitness = second_parent_fitness or 0.5

        if abs(parent1_fitness - parent2_fitness) < 0.1:
            strategy = "uniform"
            strategy_desc = "uniform crossover (similar fitness parents)"
        elif parent1_fitness > parent2_fitness:
            strategy = "biased_p1"
            strategy_desc = "biased toward parent 1 (higher fitness)"
        else:
            strategy = "biased_p2"
            strategy_desc = "biased toward parent 2 (higher fitness)"

        prompt = f"""You are performing genetic crossover of two reasoning guidelines.

Parent 1 (fitness: {parent1_fitness:.3f}):
{grg}

Parent 2 (fitness: {parent2_fitness:.3f}):
{second_parent}

Crossover Strategy: {strategy_desc}

Generate an OFFSPRING GRG that:
1. Combines the best elements from both parents
2. Resolves any conflicts between their approaches
3. Creates a coherent, unified guideline
4. {'Favors Parent 1 structure and principles' if strategy == 'biased_p1' else 'Favors Parent 2 structure and principles' if strategy == 'biased_p2' else 'Draws equally from both parents'}

Output ONLY the offspring GRG:"""

        offspring = await self.llm.complete(prompt, max_tokens=1200, temperature=0.8)
        offspring = offspring.strip()

        rationale = f"Crossover ({strategy}): P1 fitness={parent1_fitness:.3f}, P2 fitness={parent2_fitness:.3f}"

        log = self._log_mutation(
            parent_grg=grg,
            child_grg=offspring,
            rationale=rationale,
            fitness_before=parent1_fitness,
            metadata={
                "strategy": strategy,
                "parent1_fitness": parent1_fitness,
                "parent2_fitness": parent2_fitness,
                "second_parent_preview": second_parent[:200],
            },
        )

        return {
            "grg": offspring,
            "mutation_type": self.mutation_type,
            "rationale": rationale,
            "mutation_id": log.mutation_id,
            "parent_id": None,
        }


# =============================================================================
# Mutation Registry & Factory
# =============================================================================


class MutationType(Enum):
    """Enumeration of available mutation types."""

    DIRECT = "direct"
    EDA = "eda"
    HYPERMUTATION = "hypermutation_reinit"
    LAMARCKIAN = "lamarckian"
    CROSSOVER = "crossover"


MUTATION_REGISTRY: dict[MutationType, type[GRGMutation]] = {
    MutationType.DIRECT: DirectMutation,
    MutationType.EDA: EDAMutation,
    MutationType.HYPERMUTATION: HypermutationReinit,
    MutationType.LAMARCKIAN: LamarckianMutation,
    MutationType.CROSSOVER: PromptCrossover,
}


def create_mutation(
    mutation_type: MutationType | str,
    llm_client: LLMClient,
    seed: int = 42,
    log_dir: Path | None = None,
    **kwargs: Any,
) -> GRGMutation:
    """Factory function to create mutation operators.

    Args:
        mutation_type: Type of mutation to create
        llm_client: LLM client for generating mutations
        seed: Random seed
        log_dir: Directory for mutation logs
        **kwargs: Additional arguments passed to mutation constructor

    Returns:
        Configured GRGMutation instance
    """
    if isinstance(mutation_type, str):
        mutation_type = MutationType(mutation_type)

    if mutation_type not in MUTATION_REGISTRY:
        raise ValueError(f"Unknown mutation type: {mutation_type}")

    mutation_class = MUTATION_REGISTRY[mutation_type]
    return mutation_class(llm_client=llm_client, seed=seed, log_dir=log_dir, **kwargs)


class MutationOrchestrator:
    """Orchestrates multiple mutation operators with configurable weights."""

    def __init__(
        self,
        llm_client: LLMClient,
        weights: dict[MutationType, float] | None = None,
        seed: int = 42,
        log_dir: Path | None = None,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            llm_client: LLM client for mutations
            weights: Probability weights for each mutation type
            seed: Random seed
            log_dir: Directory for logs
        """
        self.llm = llm_client
        self.seed = seed
        self.rng = random.Random(seed)
        self.log_dir = log_dir

        self.weights = weights or {
            MutationType.DIRECT: 0.30,
            MutationType.EDA: 0.20,
            MutationType.HYPERMUTATION: 0.15,
            MutationType.LAMARCKIAN: 0.20,
            MutationType.CROSSOVER: 0.15,
        }

        # Create all mutation operators
        self.mutations: dict[MutationType, GRGMutation] = {}
        for mut_type in MutationType:
            self.mutations[mut_type] = create_mutation(
                mut_type, llm_client, seed, log_dir
            )

        self.logger = logging.getLogger(self.__class__.__name__)

    def select_mutation(self) -> GRGMutation:
        """Select a mutation operator based on weights."""
        types = list(self.weights.keys())
        probs = list(self.weights.values())
        selected = self.rng.choices(types, weights=probs, k=1)[0]
        return self.mutations[selected]

    async def mutate(
        self,
        grg: str,
        fitness_history: FitnessHistory,
        **kwargs: Any,
    ) -> MutationResult:
        """Apply a randomly selected mutation.

        Args:
            grg: GRG to mutate
            fitness_history: Fitness history
            **kwargs: Additional arguments passed to mutation

        Returns:
            MutationResult from the selected operator
        """
        mutation = self.select_mutation()
        self.logger.info(f"Selected mutation: {mutation.mutation_type}")
        return await mutation.mutate(grg, fitness_history, **kwargs)

    def get_all_logs(self) -> list[MutationLog]:
        """Get all mutation logs from all operators."""
        logs = []
        for mutation in self.mutations.values():
            logs.extend(mutation.mutation_logs)
        return sorted(logs, key=lambda x: x.timestamp)

    def save_all_logs(self, path: Path) -> None:
        """Save all mutation logs to a single file."""
        logs = self.get_all_logs()
        with open(path, "w") as f:
            for log in logs:
                f.write(json.dumps(log.to_dict()) + "\n")


# =============================================================================
# Utility Functions
# =============================================================================


def sample_seed_grg(
    num_modules: int = 5,
    categories: list[str] | None = None,
    seed: int | None = None,
) -> str:
    """Sample a seed GRG from atomic modules.

    Args:
        num_modules: Number of modules to include
        categories: Specific categories to sample from
        seed: Random seed

    Returns:
        A seed GRG string
    """
    rng = random.Random(seed)

    if categories:
        available = []
        for cat in categories:
            if cat in ATOMIC_MODULES_BY_CATEGORY:
                available.extend(ATOMIC_MODULES_BY_CATEGORY[cat])
    else:
        available = ATOMIC_MODULES

    selected = rng.sample(available, min(num_modules, len(available)))

    grg = "When solving problems, follow these guidelines:\n\n"
    for i, module in enumerate(selected, 1):
        grg += f"{i}. {module}\n"

    return grg


def get_atomic_modules_by_category(category: str) -> list[str]:
    """Get atomic modules for a specific category.

    Args:
        category: Category name

    Returns:
        List of modules in that category
    """
    return ATOMIC_MODULES_BY_CATEGORY.get(category, [])


def list_all_categories() -> list[str]:
    """List all available atomic module categories."""
    return list(ATOMIC_MODULES_BY_CATEGORY.keys())
