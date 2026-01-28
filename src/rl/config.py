"""Configuration dataclasses for GRPO training.

This module defines configuration dataclasses for:
- GRPOConfig: GRPO training hyperparameters
- LoRAConfig: LoRA adapter configuration
- RLConfig: Complete RL training configuration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GRPOConfig:
    """GRPO training configuration.

    Attributes:
        model_name: HuggingFace model name or path
        num_generations: Number of generations per prompt (group size)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        learning_rate: Learning rate for policy optimization
        kl_coef: KL divergence penalty coefficient
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        max_grad_norm: Maximum gradient norm for clipping
        warmup_ratio: Warmup ratio for learning rate scheduler
        weight_decay: Weight decay for optimizer
        logging_steps: Steps between logging
        save_steps: Steps between checkpoints
        eval_steps: Steps between evaluations
        bf16: Whether to use bfloat16 precision
        save_total_limit: Maximum number of checkpoints to keep
        output_dir: Output directory for checkpoints
    """

    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    num_generations: int = 4
    generation_batch_size: int = 8
    max_new_tokens: int = 512
    temperature: float = 0.7
    learning_rate: float = 1e-6
    kl_coef: float = 0.1
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    bf16: bool = True
    save_total_limit: int = 3
    output_dir: str = "results/grpo"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GRPOConfig:
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "num_generations": self.num_generations,
            "generation_batch_size": self.generation_batch_size,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "learning_rate": self.learning_rate,
            "kl_coef": self.kl_coef,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_grad_norm": self.max_grad_norm,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "bf16": self.bf16,
            "save_total_limit": self.save_total_limit,
            "output_dir": self.output_dir,
        }


@dataclass
class LoRAConfig:
    """LoRA adapter configuration.

    Attributes:
        enabled: Whether to use LoRA
        r: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        lora_dropout: Dropout probability for LoRA layers
        target_modules: List of module names to apply LoRA
        task_type: Task type for PEFT (e.g., "CAUSAL_LM")
    """

    enabled: bool = True
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    task_type: str = "CAUSAL_LM"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoRAConfig:
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "task_type": self.task_type,
        }

    def to_peft_config(self) -> Any:
        """Convert to PEFT LoraConfig."""
        from peft import LoraConfig as PeftLoraConfig, TaskType

        task_type_map = {
            "CAUSAL_LM": TaskType.CAUSAL_LM,
            "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
        }

        return PeftLoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            task_type=task_type_map.get(self.task_type, TaskType.CAUSAL_LM),
            bias="none",
        )


@dataclass
class DatasetConfig:
    """Dataset configuration.

    Attributes:
        name: HuggingFace dataset name
        train_size: Number of training samples
        val_size: Number of validation samples
        seed: Random seed
        task_types: List of task types to include
        task_weights: Weights for each task type
    """

    name: str = "openbmb/UltraInteract_sft"
    train_size: int = 5000
    val_size: int = 500
    seed: int = 42
    task_types: list[str] = field(
        default_factory=lambda: ["Coding", "Math_CoT", "Math_PoT", "Logic"]
    )
    task_weights: dict[str, float] = field(
        default_factory=lambda: {
            "Coding": 0.30,
            "Math_CoT": 0.25,
            "Math_PoT": 0.25,
            "Logic": 0.20,
        }
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetConfig:
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class RewardConfig:
    """Reward function configuration.

    Attributes:
        correct_reward: Reward for correct answer
        incorrect_reward: Reward for incorrect answer
        partial_credit: Whether to use partial credit
        math_tolerance: Tolerance for mathematical comparisons
    """

    correct_reward: float = 1.0
    incorrect_reward: float = 0.0
    partial_credit: bool = False
    math_tolerance: float = 1e-4

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RewardConfig:
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class EvaluationConfig:
    """Evaluation configuration.

    Attributes:
        batch_size: Batch size for evaluation
        max_samples: Maximum samples for evaluation
        metrics: List of metrics to compute
    """

    batch_size: int = 8
    max_samples: int = 500
    metrics: list[str] = field(default_factory=lambda: ["accuracy", "per_task_accuracy"])

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationConfig:
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class WandBConfig:
    """Weights & Biases configuration.

    Attributes:
        enabled: Whether to enable WandB logging
        project: WandB project name
        entity: WandB entity (username or team)
        tags: List of tags for the run
    """

    enabled: bool = False
    project: str = "lore-rl"
    entity: str | None = None
    tags: list[str] = field(default_factory=lambda: ["grpo", "ultrainteract"])

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WandBConfig:
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class RLConfig:
    """Complete RL training configuration.

    Attributes:
        grpo: GRPO training configuration
        lora: LoRA adapter configuration
        dataset: Dataset configuration
        reward: Reward function configuration
        evaluation: Evaluation configuration
        wandb: WandB configuration
    """

    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RLConfig:
        """Create from dictionary."""
        return cls(
            grpo=GRPOConfig.from_dict(data.get("grpo", {})),
            lora=LoRAConfig.from_dict(data.get("lora", {})),
            dataset=DatasetConfig.from_dict(data.get("dataset", {})),
            reward=RewardConfig.from_dict(data.get("reward", {})),
            evaluation=EvaluationConfig.from_dict(data.get("evaluation", {})),
            wandb=WandBConfig.from_dict(data.get("wandb", {})),
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> RLConfig:
        """Load from YAML file."""
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "grpo": self.grpo.to_dict(),
            "lora": self.lora.to_dict(),
            "dataset": {
                "name": self.dataset.name,
                "train_size": self.dataset.train_size,
                "val_size": self.dataset.val_size,
                "seed": self.dataset.seed,
                "task_types": self.dataset.task_types,
                "task_weights": self.dataset.task_weights,
            },
            "reward": {
                "correct_reward": self.reward.correct_reward,
                "incorrect_reward": self.reward.incorrect_reward,
                "partial_credit": self.reward.partial_credit,
                "math_tolerance": self.reward.math_tolerance,
            },
            "evaluation": {
                "batch_size": self.evaluation.batch_size,
                "max_samples": self.evaluation.max_samples,
                "metrics": self.evaluation.metrics,
            },
            "wandb": {
                "enabled": self.wandb.enabled,
                "project": self.wandb.project,
                "entity": self.wandb.entity,
                "tags": self.wandb.tags,
            },
        }

    def save(self, path: Path | str) -> None:
        """Save to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def load_config(path: Path | str | None = None) -> RLConfig:
    """Load RL configuration from file or use defaults.

    Args:
        path: Path to YAML config file. If None, uses default config.

    Returns:
        RLConfig object
    """
    if path is None:
        return RLConfig()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    return RLConfig.from_yaml(path)
