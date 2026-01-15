"""PromptBreeder: Self-referential prompt evolution.

Implements the PromptBreeder algorithm which evolves both task prompts
and mutation prompts (meta-prompts) simultaneously.

Reference: Fernando et al., "PromptBreeder: Self-Referential Self-Improvement
Via Prompt Evolution" (2023)
"""

import asyncio
import random
import time
from typing import Any, Callable

from src.models.frontier_api import FrontierAPIClient
from src.optimization.base_optimizer import (
    BaseOptimizer,
    OptimizationResult,
    PromptIndividual,
)


class MutationPrompt(PromptIndividual):
    """A meta-prompt that generates variations of task prompts."""

    mutation_type: str = "thinking_style"


class PromptUnit:
    """A unit containing both a task prompt and its associated mutation prompt."""

    def __init__(
        self,
        task_prompt: PromptIndividual,
        mutation_prompt: MutationPrompt,
    ) -> None:
        self.task_prompt = task_prompt
        self.mutation_prompt = mutation_prompt

    @property
    def fitness(self) -> float:
        return self.task_prompt.fitness


class PromptBreederOptimizer(BaseOptimizer):
    """PromptBreeder optimizer with self-referential improvement.

    Key features:
    - Evolves both task prompts and mutation prompts
    - Multiple mutation operators including hyper-mutations
    - Diversity maintenance through semantic clustering
    - Self-referential improvement of mutation operators
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.evolution_model: FrontierAPIClient | None = None
        self.mutation_prompts: list[MutationPrompt] = []

    def _default_config(self) -> dict[str, Any]:
        return {
            "population": {
                "size": 50,
                "elite_size": 5,
            },
            "evolution": {
                "generations": 100,
            },
            "promptbreeder": {
                "thinking_styles": {
                    "initial_pool": [
                        "Let me break this down step by step",
                        "First, I'll identify the key components",
                        "To solve this, I need to consider",
                    ]
                },
                "mutation_prompts": [
                    "Rephrase the following instruction to be more precise: {prompt}",
                    "Make this instruction more concise: {prompt}",
                    "Add step-by-step structure to: {prompt}",
                ],
                "self_improvement": {
                    "enabled": True,
                    "generations_between_meta_updates": 10,
                },
            },
            "diversity": {
                "method": "semantic_clustering",
                "min_clusters": 5,
                "similarity_threshold": 0.85,
            },
            "early_stopping": {
                "enabled": True,
                "patience": 20,
                "min_delta": 0.001,
            },
        }

    def _validate_config(self) -> None:
        if "promptbreeder" not in self.config:
            raise ValueError("Missing 'promptbreeder' config section")

    def set_evolution_model(self, client: FrontierAPIClient) -> None:
        """Set the model client for prompt evolution."""
        self.evolution_model = client

    def _initialize_mutation_prompts(self) -> list[MutationPrompt]:
        """Initialize the pool of mutation prompts."""
        pb_config = self.config["promptbreeder"]
        prompts = pb_config.get("mutation_prompts", [])

        return [
            MutationPrompt(content=p, mutation_type="direct") for p in prompts
        ]

    def initialize_population(
        self, seed_prompts: list[str] | None = None
    ) -> list[PromptIndividual]:
        """Initialize population with thinking-style prompts."""
        pb_config = self.config["promptbreeder"]
        pop_size = self.config["population"]["size"]

        # Initialize mutation prompts
        self.mutation_prompts = self._initialize_mutation_prompts()

        # Get thinking styles
        thinking_styles = pb_config.get("thinking_styles", {}).get("initial_pool", [])

        if seed_prompts:
            base_prompts = seed_prompts
        else:
            base_prompts = thinking_styles or [
                "Let me solve this step by step.",
                "I'll analyze this problem carefully.",
            ]

        population = []
        for i in range(pop_size):
            content = base_prompts[i % len(base_prompts)]
            population.append(PromptIndividual(content=content, generation=0))

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

    async def _direct_mutation(
        self,
        individual: PromptIndividual,
        mutation_prompt: MutationPrompt,
        generation: int,
    ) -> PromptIndividual:
        """Apply a mutation prompt to generate a variation."""
        if self.evolution_model is None:
            raise RuntimeError("Evolution model not set.")

        prompt = mutation_prompt.content.format(prompt=individual.content)

        response = await self.evolution_model.complete(
            prompt=prompt,
            max_tokens=500,
            temperature=0.8,
        )

        return PromptIndividual(
            content=response.strip(),
            generation=generation,
            parent_ids=[individual.id],
            mutation_history=individual.mutation_history + ["direct_mutation"],
        )

    async def _eda_mutation(
        self,
        individuals: list[PromptIndividual],
        generation: int,
    ) -> PromptIndividual:
        """Estimation of Distribution Algorithm mutation.

        Generates new prompt based on distribution of high-fitness prompts.
        """
        if self.evolution_model is None:
            raise RuntimeError("Evolution model not set.")

        # Select top performers
        top_k = sorted(individuals, key=lambda x: x.fitness, reverse=True)[:5]
        examples = "\n".join([f"- {ind.content}" for ind in top_k])

        prompt = f"""Here are several high-quality instruction prompts:
{examples}

Generate a new instruction prompt that captures the best qualities of these examples while being distinct:"""

        response = await self.evolution_model.complete(
            prompt=prompt,
            max_tokens=500,
            temperature=0.9,
        )

        return PromptIndividual(
            content=response.strip(),
            generation=generation,
            mutation_history=["eda_mutation"],
        )

    async def _hyper_mutation(
        self,
        mutation_prompt: MutationPrompt,
    ) -> MutationPrompt:
        """Mutate a mutation prompt (meta-mutation)."""
        if self.evolution_model is None:
            raise RuntimeError("Evolution model not set.")

        prompt = f"""Improve the following prompt-mutation instruction to make it more effective at generating diverse, high-quality variations:

Original: {mutation_prompt.content}

Improved version:"""

        response = await self.evolution_model.complete(
            prompt=prompt,
            max_tokens=300,
            temperature=0.7,
        )

        return MutationPrompt(
            content=response.strip(),
            mutation_type="hyper_mutated",
            parent_ids=[mutation_prompt.id],
        )

    async def _update_mutation_prompts(self) -> None:
        """Self-referentially improve mutation prompts."""
        if not self.mutation_prompts:
            return

        # Hyper-mutate a random subset
        num_to_mutate = max(1, len(self.mutation_prompts) // 3)
        to_mutate = random.sample(self.mutation_prompts, num_to_mutate)

        tasks = [self._hyper_mutation(mp) for mp in to_mutate]
        new_prompts = await asyncio.gather(*tasks, return_exceptions=True)

        for new_mp in new_prompts:
            if isinstance(new_mp, MutationPrompt):
                self.mutation_prompts.append(new_mp)

        # Keep pool size manageable
        if len(self.mutation_prompts) > 20:
            self.mutation_prompts = self.mutation_prompts[-20:]

    async def evolve_generation(
        self,
        population: list[PromptIndividual],
        generation: int,
    ) -> list[PromptIndividual]:
        """Evolve population using PromptBreeder operators."""
        pop_config = self.config["population"]
        pb_config = self.config["promptbreeder"]

        # Preserve elites
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        elite_size = pop_config["elite_size"]
        new_population = list(sorted_pop[:elite_size])

        # Self-improvement of mutation prompts
        si_config = pb_config.get("self_improvement", {})
        if si_config.get("enabled", False):
            interval = si_config.get("generations_between_meta_updates", 10)
            if generation % interval == 0:
                await self._update_mutation_prompts()

        # Generate offspring
        target_size = pop_config["size"]
        tasks = []

        while len(new_population) + len(tasks) < target_size:
            op_choice = random.random()

            if op_choice < 0.5 and self.mutation_prompts:
                # Direct mutation
                parent = random.choice(sorted_pop[:20])
                mp = random.choice(self.mutation_prompts)
                tasks.append(self._direct_mutation(parent, mp, generation))
            elif op_choice < 0.8:
                # EDA mutation
                tasks.append(self._eda_mutation(sorted_pop, generation))
            else:
                # Random from top performers
                parent = random.choice(sorted_pop[:10])
                new_population.append(
                    PromptIndividual(
                        content=parent.content,
                        generation=generation,
                        parent_ids=[parent.id],
                    )
                )

        offspring = await asyncio.gather(*tasks, return_exceptions=True)
        for child in offspring:
            if isinstance(child, PromptIndividual):
                new_population.append(child)

        return new_population[: pop_config["size"]]

    async def optimize(
        self,
        eval_fn: Callable[[str], float],
        seed_prompts: list[str] | None = None,
    ) -> OptimizationResult:
        """Run the full PromptBreeder optimization loop."""
        start_time = time.time()
        generations = self.config["evolution"]["generations"]

        population = self.initialize_population(seed_prompts)
        fitness_history: list[float] = []
        total_evaluations = 0

        for gen in range(generations):
            # Evaluate
            eval_tasks = [self.evaluate_fitness(ind, eval_fn) for ind in population]
            await asyncio.gather(*eval_tasks)
            total_evaluations += len(population)

            # Track progress
            best = max(population, key=lambda x: x.fitness)
            fitness_history.append(best.fitness)
            self.log_generation(gen, population, best.fitness)

            # Early stopping
            if self.should_stop_early(fitness_history):
                self.logger.info(f"Early stopping at generation {gen}")
                break

            # Evolve
            if gen < generations - 1:
                population = await self.evolve_generation(population, gen + 1)

        wall_time = time.time() - start_time
        best = max(population, key=lambda x: x.fitness)

        return OptimizationResult(
            best_prompt=best,
            final_population=population,
            fitness_history=fitness_history,
            generations_completed=len(fitness_history),
            total_evaluations=total_evaluations,
            wall_time_seconds=wall_time,
            config=self.config,
            metadata={
                "optimizer": "PromptBreeder",
                "final_mutation_prompts": [mp.content for mp in self.mutation_prompts],
            },
        )
