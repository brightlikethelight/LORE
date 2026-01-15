"""Base class for prompt optimization methods."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml
from pydantic import BaseModel


class PromptIndividual(BaseModel):
    """Represents a single prompt in the optimization population."""

    content: str
    fitness: float = 0.0
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    mutation_history: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Generate unique ID based on content hash."""
        import hashlib

        return hashlib.sha256(self.content.encode()).hexdigest()[:12]


@dataclass
class OptimizationResult:
    """Results from a prompt optimization run."""

    best_prompt: PromptIndividual
    final_population: list[PromptIndividual]
    fitness_history: list[float]
    generations_completed: int
    total_evaluations: int
    wall_time_seconds: float
    config: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        """Save optimization results to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save best prompt
        (path / "best_prompt.txt").write_text(self.best_prompt.content)

        # Save full results as YAML
        results_dict = {
            "best_prompt": self.best_prompt.model_dump(),
            "fitness_history": self.fitness_history,
            "generations_completed": self.generations_completed,
            "total_evaluations": self.total_evaluations,
            "wall_time_seconds": self.wall_time_seconds,
            "config": self.config,
            "metadata": self.metadata,
        }
        (path / "results.yaml").write_text(yaml.dump(results_dict, default_flow_style=False))

    @classmethod
    def load(cls, path: Path) -> "OptimizationResult":
        """Load optimization results from disk."""
        path = Path(path)
        with open(path / "results.yaml") as f:
            data = yaml.safe_load(f)

        return cls(
            best_prompt=PromptIndividual(**data["best_prompt"]),
            final_population=[],  # Not saved by default
            fitness_history=data["fitness_history"],
            generations_completed=data["generations_completed"],
            total_evaluations=data["total_evaluations"],
            wall_time_seconds=data["wall_time_seconds"],
            config=data["config"],
            metadata=data.get("metadata", {}),
        )


class BaseOptimizer(ABC):
    """Abstract base class for prompt optimization methods.

    All optimization methods (GEPA, PromptBreeder, MIPRO) inherit from this
    class and implement the core optimization loop.
    """

    def __init__(
        self,
        config_path: Path | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the optimizer.

        Args:
            config_path: Path to YAML configuration file.
            config: Configuration dictionary (overrides config_path).
        """
        if config is not None:
            self.config = config
        elif config_path is not None:
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()

        self._validate_config()
        self._setup_logging()

    @abstractmethod
    def _default_config(self) -> dict[str, Any]:
        """Return default configuration for this optimizer."""
        ...

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate the configuration."""
        ...

    def _setup_logging(self) -> None:
        """Set up logging and experiment tracking."""
        import logging

        self.logger = logging.getLogger(self.__class__.__name__)

        if self.config.get("logging", {}).get("wandb", {}).get("enabled", False):
            import wandb

            wandb_config = self.config["logging"]["wandb"]
            wandb.init(
                project=wandb_config.get("project", "lore-optimization"),
                entity=wandb_config.get("entity"),
                tags=wandb_config.get("tags", []),
                config=self.config,
            )
            self.use_wandb = True
        else:
            self.use_wandb = False

    @abstractmethod
    def initialize_population(self, seed_prompts: list[str] | None = None) -> list[PromptIndividual]:
        """Initialize the population of prompts.

        Args:
            seed_prompts: Optional list of prompts to seed the population.

        Returns:
            Initial population of PromptIndividuals.
        """
        ...

    @abstractmethod
    async def evaluate_fitness(
        self,
        individual: PromptIndividual,
        eval_fn: Callable[[str], float],
    ) -> float:
        """Evaluate the fitness of a prompt individual.

        Args:
            individual: The prompt to evaluate.
            eval_fn: Function that takes a prompt and returns a fitness score.

        Returns:
            Fitness score for the individual.
        """
        ...

    @abstractmethod
    async def evolve_generation(
        self,
        population: list[PromptIndividual],
        generation: int,
    ) -> list[PromptIndividual]:
        """Evolve the population for one generation.

        Args:
            population: Current population.
            generation: Current generation number.

        Returns:
            New population after evolution.
        """
        ...

    @abstractmethod
    async def optimize(
        self,
        eval_fn: Callable[[str], float],
        seed_prompts: list[str] | None = None,
    ) -> OptimizationResult:
        """Run the full optimization loop.

        Args:
            eval_fn: Function to evaluate prompt fitness.
            seed_prompts: Optional seed prompts for initialization.

        Returns:
            OptimizationResult containing the best prompt and metadata.
        """
        ...

    def should_stop_early(self, fitness_history: list[float]) -> bool:
        """Check if early stopping criteria are met.

        Args:
            fitness_history: List of best fitness values per generation.

        Returns:
            True if optimization should stop early.
        """
        early_stop_config = self.config.get("early_stopping", {})
        if not early_stop_config.get("enabled", False):
            return False

        patience = early_stop_config.get("patience", 10)
        min_delta = early_stop_config.get("min_delta", 0.001)

        if len(fitness_history) < patience:
            return False

        recent = fitness_history[-patience:]
        improvement = max(recent) - min(recent)

        return improvement < min_delta

    def log_generation(
        self,
        generation: int,
        population: list[PromptIndividual],
        best_fitness: float,
    ) -> None:
        """Log generation statistics.

        Args:
            generation: Current generation number.
            population: Current population.
            best_fitness: Best fitness in current generation.
        """
        fitnesses = [ind.fitness for ind in population]
        avg_fitness = sum(fitnesses) / len(fitnesses)

        self.logger.info(
            f"Generation {generation}: best={best_fitness:.4f}, avg={avg_fitness:.4f}"
        )

        if self.use_wandb:
            import wandb

            wandb.log(
                {
                    "generation": generation,
                    "best_fitness": best_fitness,
                    "avg_fitness": avg_fitness,
                    "min_fitness": min(fitnesses),
                    "max_fitness": max(fitnesses),
                    "population_size": len(population),
                }
            )
