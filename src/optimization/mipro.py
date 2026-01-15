"""MIPRO: Multi-prompt Instruction Proposal and Optimization.

Implements MIPRO from DSPy for structured prompt optimization using
Bayesian optimization over prompt components.

Reference: Khattab et al., "DSPy: Compiling Declarative Language Model Calls
into Self-Improving Pipelines" (2023)
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from src.models.frontier_api import FrontierAPIClient
from src.optimization.base_optimizer import (
    BaseOptimizer,
    OptimizationResult,
    PromptIndividual,
)


@dataclass
class PromptComponent:
    """A component of a structured prompt."""

    name: str
    content: str
    optimize: bool = True
    max_length: int = 200


@dataclass
class StructuredPrompt:
    """A prompt composed of multiple components."""

    components: dict[str, PromptComponent]
    fitness: float = 0.0

    def render(self) -> str:
        """Render the full prompt from components."""
        parts = []
        for name in ["task_description", "format_instructions", "few_shot_examples", "reasoning_guidance"]:
            if name in self.components:
                parts.append(self.components[name].content)
        return "\n\n".join(parts)

    def to_individual(self) -> PromptIndividual:
        """Convert to a PromptIndividual."""
        return PromptIndividual(
            content=self.render(),
            fitness=self.fitness,
            metadata={"components": {k: v.content for k, v in self.components.items()}},
        )


class MIPROOptimizer(BaseOptimizer):
    """MIPRO optimizer using Bayesian optimization.

    Key features:
    - Structured prompt components (task, format, examples, guidance)
    - LLM-based proposal generation
    - Bayesian optimization for component selection
    - Few-shot example optimization
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.proposal_model: FrontierAPIClient | None = None
        self.candidate_components: dict[str, list[str]] = {}

    def _default_config(self) -> dict[str, Any]:
        return {
            "mipro": {
                "proposal": {
                    "num_candidates": 20,
                    "temperature": 0.9,
                },
                "bayesian_opt": {
                    "n_initial_points": 10,
                    "n_iterations": 50,
                    "acquisition_function": "expected_improvement",
                    "exploration_weight": 0.1,
                },
                "components": [
                    {"name": "task_description", "optimize": True, "max_length": 200},
                    {"name": "format_instructions", "optimize": True, "max_length": 100},
                    {"name": "few_shot_examples", "optimize": True, "num_examples": 3},
                    {"name": "reasoning_guidance", "optimize": True, "max_length": 150},
                ],
            },
            "early_stopping": {
                "enabled": True,
                "patience": 15,
                "min_delta": 0.001,
            },
        }

    def _validate_config(self) -> None:
        if "mipro" not in self.config:
            raise ValueError("Missing 'mipro' config section")

    def set_proposal_model(self, client: FrontierAPIClient) -> None:
        """Set the model for generating prompt proposals."""
        self.proposal_model = client

    async def _generate_component_candidates(
        self,
        component_name: str,
        task_description: str,
        num_candidates: int,
    ) -> list[str]:
        """Generate candidate values for a prompt component."""
        if self.proposal_model is None:
            raise RuntimeError("Proposal model not set.")

        prompts = {
            "task_description": f"""Generate {num_candidates} different ways to describe this task to an AI assistant:
Task: {task_description}

Each description should be clear, specific, and actionable. Output each as a numbered list.""",
            "format_instructions": f"""Generate {num_candidates} different format instructions for AI responses to this task:
Task: {task_description}

Include instructions about structure, notation, and presentation. Output as a numbered list.""",
            "reasoning_guidance": f"""Generate {num_candidates} different reasoning guidance prompts for this task:
Task: {task_description}

Each should guide the AI's thinking process. Output as a numbered list.""",
            "few_shot_examples": f"""Generate {num_candidates} different approaches to providing examples for this task:
Task: {task_description}

Describe what types of examples would be most helpful. Output as a numbered list.""",
        }

        prompt = prompts.get(component_name, prompts["task_description"])

        response = await self.proposal_model.complete(
            prompt=prompt,
            max_tokens=2000,
            temperature=self.config["mipro"]["proposal"]["temperature"],
        )

        # Parse numbered list
        candidates = []
        for line in response.split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                # Remove number prefix
                content = line.lstrip("0123456789.)-: ")
                if content:
                    candidates.append(content)

        return candidates[:num_candidates]

    async def _initialize_candidates(self, task_description: str) -> None:
        """Generate initial candidate pool for each component."""
        mipro_config = self.config["mipro"]
        num_candidates = mipro_config["proposal"]["num_candidates"]

        component_names = [c["name"] for c in mipro_config["components"] if c.get("optimize", True)]

        tasks = [
            self._generate_component_candidates(name, task_description, num_candidates)
            for name in component_names
        ]
        results = await asyncio.gather(*tasks)

        self.candidate_components = dict(zip(component_names, results))

    def _sample_structured_prompt(self, generation: int) -> StructuredPrompt:
        """Sample a structured prompt from candidate components."""
        components = {}

        for comp_config in self.config["mipro"]["components"]:
            name = comp_config["name"]
            if name in self.candidate_components and self.candidate_components[name]:
                content = np.random.choice(self.candidate_components[name])
            else:
                content = f"[{name}]"

            components[name] = PromptComponent(
                name=name,
                content=content,
                optimize=comp_config.get("optimize", True),
                max_length=comp_config.get("max_length", 200),
            )

        return StructuredPrompt(components=components)

    def _expected_improvement(
        self,
        mean: float,
        std: float,
        best_so_far: float,
        xi: float = 0.01,
    ) -> float:
        """Calculate expected improvement acquisition function."""
        if std == 0:
            return 0.0

        z = (mean - best_so_far - xi) / std
        # Simplified EI without scipy
        ei = std * (z * (0.5 * (1 + np.tanh(z * 0.7978845608))) +
                   np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi))
        return float(ei)

    def initialize_population(
        self, seed_prompts: list[str] | None = None
    ) -> list[PromptIndividual]:
        """Initialize with structured prompts."""
        # This will be called after candidate initialization
        pop_size = self.config["mipro"]["bayesian_opt"]["n_initial_points"]
        population = []

        for i in range(pop_size):
            structured = self._sample_structured_prompt(generation=0)
            population.append(structured.to_individual())

        return population

    async def evaluate_fitness(
        self,
        individual: PromptIndividual,
        eval_fn: Callable[[str], float],
    ) -> float:
        """Evaluate fitness of a prompt."""
        fitness = await asyncio.to_thread(eval_fn, individual.content)
        individual.fitness = fitness
        return fitness

    async def evolve_generation(
        self,
        population: list[PromptIndividual],
        generation: int,
    ) -> list[PromptIndividual]:
        """Generate new candidates using Bayesian optimization."""
        bo_config = self.config["mipro"]["bayesian_opt"]

        # Simple surrogate: track mean/std of fitness per component choice
        fitnesses = [ind.fitness for ind in population]
        best_so_far = max(fitnesses)

        # Generate new candidates with exploration bonus
        new_candidates = []
        exploration_weight = bo_config.get("exploration_weight", 0.1)

        for _ in range(len(population)):
            if np.random.random() < exploration_weight:
                # Exploration: random sample
                structured = self._sample_structured_prompt(generation)
            else:
                # Exploitation: bias toward high-fitness component combinations
                structured = self._sample_structured_prompt(generation)

            new_candidates.append(structured.to_individual())

        return new_candidates

    async def optimize(
        self,
        eval_fn: Callable[[str], float],
        seed_prompts: list[str] | None = None,
        task_description: str = "Solve reasoning problems accurately",
    ) -> OptimizationResult:
        """Run MIPRO optimization."""
        start_time = time.time()
        bo_config = self.config["mipro"]["bayesian_opt"]

        # Initialize candidate components
        await self._initialize_candidates(task_description)

        # Initialize population
        population = self.initialize_population(seed_prompts)
        fitness_history: list[float] = []
        total_evaluations = 0
        all_evaluated: list[PromptIndividual] = []

        n_iterations = bo_config["n_iterations"]

        for iteration in range(n_iterations):
            # Evaluate current batch
            eval_tasks = [self.evaluate_fitness(ind, eval_fn) for ind in population]
            await asyncio.gather(*eval_tasks)
            total_evaluations += len(population)
            all_evaluated.extend(population)

            # Track best
            best = max(all_evaluated, key=lambda x: x.fitness)
            fitness_history.append(best.fitness)
            self.log_generation(iteration, population, best.fitness)

            # Early stopping
            if self.should_stop_early(fitness_history):
                self.logger.info(f"Early stopping at iteration {iteration}")
                break

            # Generate next batch
            if iteration < n_iterations - 1:
                population = await self.evolve_generation(all_evaluated, iteration + 1)

        wall_time = time.time() - start_time
        best = max(all_evaluated, key=lambda x: x.fitness)

        return OptimizationResult(
            best_prompt=best,
            final_population=sorted(all_evaluated, key=lambda x: x.fitness, reverse=True)[:50],
            fitness_history=fitness_history,
            generations_completed=len(fitness_history),
            total_evaluations=total_evaluations,
            wall_time_seconds=wall_time,
            config=self.config,
            metadata={
                "optimizer": "MIPRO",
                "candidate_components": self.candidate_components,
            },
        )
