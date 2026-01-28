"""SkyRL GRPO Trainer for LORE.

This module provides a trainer implementation using the SkyRL framework
for GRPO (Group Relative Policy Optimization) training.

SkyRL (from UC Berkeley's Sky Computing Lab) is a modular RL library
designed for multi-turn agentic and long-horizon reasoning tasks.
It offers faster single-GPU training compared to Verl due to:
- Better async generation vs serial fallback
- Simpler setup without mandatory Ray cluster
- Native GRPO support with batched generation

Reference: https://github.com/skypilot-org/skyrl

NOTE: The skyrl package may not be published to PyPI yet (as of 2025).
This implementation provides a fallback that works without the SkyRL
library installed, using optimized batched generation with vLLM or
HuggingFace transformers.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import torch
from datasets import Dataset

logger = logging.getLogger(__name__)


@dataclass
class SkyRLConfig:
    """Configuration for SkyRL GRPO training.

    Attributes:
        model_name: HuggingFace model name or path
        group_size: Number of generations per prompt (GRPO group size)
        learning_rate: Learning rate for policy optimization
        clip_ratio: Clip ratio for GRPO policy loss
        beta: KL penalty coefficient
        num_epochs: Number of training epochs
        batch_size: Number of prompts per batch
        max_prompt_length: Maximum prompt length
        max_response_length: Maximum response length
        temperature: Sampling temperature for generation
        top_p: Top-p (nucleus) sampling parameter
        gradient_accumulation_steps: Gradient accumulation steps
        max_grad_norm: Maximum gradient norm for clipping
        warmup_ratio: Warmup ratio for learning rate scheduler
        weight_decay: Weight decay for optimizer
        bf16: Whether to use bfloat16 precision
        use_lora: Whether to use LoRA for efficient fine-tuning
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        output_dir: Output directory for checkpoints
        logging_steps: Steps between logging
        save_steps: Steps between checkpoints
        eval_steps: Steps between evaluations
        save_total_limit: Maximum checkpoints to keep
        use_vllm: Whether to use vLLM for faster generation
        vllm_tensor_parallel: Tensor parallel size for vLLM
        load_in_8bit: Whether to use 8-bit quantization (disable for smaller models)
        use_gradient_checkpointing: Whether to use gradient checkpointing (disable for debugging)
    """

    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    group_size: int = 4
    learning_rate: float = 1e-6
    clip_ratio: float = 0.2
    beta: float = 0.04
    num_epochs: int = 1
    batch_size: int = 8
    max_prompt_length: int = 2048
    max_response_length: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    bf16: bool = True
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    output_dir: str = "results/skyrl_grpo"
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    use_vllm: bool = True
    vllm_tensor_parallel: int = 1
    load_in_8bit: bool = True
    use_gradient_checkpointing: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkyRLConfig:
        """Create from dictionary with type coercion."""
        filtered = {}
        for k, v in data.items():
            if k in cls.__dataclass_fields__:
                field_type = cls.__dataclass_fields__[k].type
                # Handle string values that should be floats (e.g., "5e-6")
                # Note: with `from __future__ import annotations`, types are strings
                if field_type in (float, "float") and isinstance(v, str):
                    v = float(v)
                elif field_type in (int, "int") and isinstance(v, str):
                    v = int(v)
                filtered[k] = v
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: Path | str) -> SkyRLConfig:
        """Load from YAML file.

        Handles nested YAML structure with model, grpo, lora, vllm sections.
        """
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)

        # Flatten nested structure
        flat_data: dict[str, Any] = {}

        # Extract from grpo section
        if "grpo" in data:
            flat_data.update(data["grpo"])

        # Extract model name
        if "model" in data and "name" in data["model"]:
            flat_data["model_name"] = data["model"]["name"]

        # Extract LoRA settings
        if "lora" in data:
            lora = data["lora"]
            flat_data["use_lora"] = lora.get("enabled", True)
            if "r" in lora:
                flat_data["lora_r"] = lora["r"]
            if "alpha" in lora:
                flat_data["lora_alpha"] = lora["alpha"]
            if "dropout" in lora:
                flat_data["lora_dropout"] = lora["dropout"]

        # Extract vLLM settings
        if "vllm" in data:
            vllm = data["vllm"]
            flat_data["use_vllm"] = vllm.get("enabled", True)
            if "tensor_parallel_size" in vllm:
                flat_data["vllm_tensor_parallel"] = vllm["tensor_parallel_size"]

        # Extract output directory
        if "output_dir" in data:
            flat_data["output_dir"] = data["output_dir"]

        # Extract logging/checkpoint settings
        for key in ["logging_steps", "save_steps", "eval_steps", "save_total_limit"]:
            if key in data:
                flat_data[key] = data[key]

        return cls.from_dict(flat_data)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }


@dataclass
class SkyRLTrainingResult:
    """Result of SkyRL GRPO training.

    Attributes:
        model_path: Path to saved model checkpoint
        final_loss: Final training loss
        training_steps: Total number of training steps
        wall_time_seconds: Total training time
        metrics: Training metrics history
        config: Configuration used for training
    """

    model_path: Path
    final_loss: float
    training_steps: int
    wall_time_seconds: float
    metrics: dict[str, Any]
    config: dict[str, Any]

    def save(self, path: Path | str) -> None:
        """Save training result to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        result_dict = {
            "model_path": str(self.model_path),
            "final_loss": self.final_loss,
            "training_steps": self.training_steps,
            "wall_time_seconds": self.wall_time_seconds,
            "metrics": self.metrics,
            "config": self.config,
            "timestamp": datetime.now().isoformat(),
        }

        with open(path / "training_result.json", "w") as f:
            json.dump(result_dict, f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> SkyRLTrainingResult:
        """Load training result from disk."""
        path = Path(path)
        with open(path / "training_result.json") as f:
            data = json.load(f)

        return cls(
            model_path=Path(data["model_path"]),
            final_loss=data["final_loss"],
            training_steps=data["training_steps"],
            wall_time_seconds=data["wall_time_seconds"],
            metrics=data["metrics"],
            config=data["config"],
        )


class LORESkyRLTrainer:
    """SkyRL-based GRPO Trainer for LORE experiments.

    This trainer uses the SkyRL framework for GRPO training on
    multi-task reasoning datasets (math, logic, code).

    SkyRL provides faster training compared to Verl on single GPU setups
    due to its async batched generation capabilities.

    Example usage:
        config = SkyRLConfig(model_name="Qwen/Qwen2.5-7B-Instruct")
        trainer = LORESkyRLTrainer(config)

        # Load data
        from src.rl.skyrl_data_pipeline import load_skyrl_datasets
        datasets = load_skyrl_datasets(tokenizer=trainer.tokenizer)

        # Train
        result = trainer.train(
            train_dataset=datasets["train"],
            eval_dataset=datasets["val"],
        )
    """

    def __init__(
        self,
        config: SkyRLConfig | None = None,
        reward_fn: Callable[[list[str], list[dict[str, Any]]], list[float]] | None = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            config: GRPO training configuration
            reward_fn: Reward function for computing rewards
        """
        self.config = config or SkyRLConfig()
        self._reward_fn = reward_fn

        # Lazy initialization
        self._model = None
        self._tokenizer = None
        self._vllm_model = None
        self._skyrl_trainer = None
        self._trl_sample_metadata: dict[str, dict[str, Any]] = {}  # For TRL reward function

        # Metrics tracking
        self._metrics_history: dict[str, list[float]] = {
            "loss": [],
            "reward_mean": [],
            "reward_std": [],
            "accuracy": [],
            "kl_divergence": [],
        }

    @property
    def tokenizer(self) -> Any:
        """Get the tokenizer, loading if necessary."""
        if self._tokenizer is None:
            self._load_tokenizer()
        return self._tokenizer

    @property
    def model(self) -> Any:
        """Get the model, loading if necessary."""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def reward_fn(self) -> Callable:
        """Get the reward function."""
        if self._reward_fn is None:
            from src.rl.skyrl_rewards import create_skyrl_reward_fn
            self._reward_fn = create_skyrl_reward_fn()
        return self._reward_fn

    def _load_tokenizer(self) -> None:
        """Load the tokenizer."""
        from transformers import AutoTokenizer

        logger.info(f"Loading tokenizer: {self.config.model_name}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )

        # Ensure pad token is set
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

    def _load_model(self) -> None:
        """Load the model with optional LoRA."""
        from transformers import AutoModelForCausalLM

        logger.info(f"Loading model: {self.config.model_name}")

        # Determine device map and dtype
        device_map = "auto" if torch.cuda.is_available() else None
        dtype = torch.bfloat16 if self.config.bf16 and torch.cuda.is_available() else torch.float32

        # Try to use 8-bit quantization for memory efficiency (if enabled)
        if self.config.load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    device_map=device_map,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                    quantization_config=quantization_config,
                )
                logger.info("Model loaded with 8-bit quantization")
            except (ImportError, Exception) as e:
                logger.warning(f"8-bit quantization not available ({e}), loading full precision")
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    device_map=device_map,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                )
        else:
            # Load without quantization (more stable, requires more memory)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map=device_map,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
            logger.info("Model loaded without quantization")

        # Enable gradient checkpointing to reduce memory (if enabled)
        if self.config.use_gradient_checkpointing and hasattr(self._model, "gradient_checkpointing_enable"):
            self._model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        else:
            logger.info("Gradient checkpointing disabled")

        # Apply LoRA if enabled
        if self.config.use_lora:
            self._apply_lora()

    def _apply_lora(self) -> None:
        """Apply LoRA adapter to the model."""
        from peft import LoraConfig, TaskType, get_peft_model

        logger.info("Applying LoRA adapter")

        # Prepare model for k-bit training ONLY if using quantization
        # This is required for proper gradient flow with LoRA + quantization
        if self.config.load_in_8bit:
            try:
                from peft import prepare_model_for_kbit_training
                self._model = prepare_model_for_kbit_training(
                    self._model,
                    use_gradient_checkpointing=True,
                )
                logger.info("Model prepared for k-bit training")
            except ImportError:
                logger.warning("prepare_model_for_kbit_training not available, skipping")

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=[
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )

        self._model = get_peft_model(self._model, lora_config)
        self._model.print_trainable_parameters()

    def _init_vllm(self) -> Any:
        """Initialize vLLM for faster generation.

        Returns:
            vLLM LLM instance or None if not available
        """
        if not self.config.use_vllm:
            return None

        try:
            from vllm import LLM, SamplingParams

            logger.info(f"Initializing vLLM with model: {self.config.model_name}")

            self._vllm_model = LLM(
                model=self.config.model_name,
                tensor_parallel_size=self.config.vllm_tensor_parallel,
                dtype="bfloat16" if self.config.bf16 else "float16",
                trust_remote_code=True,
                max_model_len=self.config.max_prompt_length + self.config.max_response_length,
            )

            logger.info("vLLM initialized successfully")
            return self._vllm_model

        except ImportError:
            logger.warning("vLLM not available, falling back to HuggingFace generation")
            return None
        except Exception as e:
            logger.warning(f"Failed to initialize vLLM ({e}), falling back to HuggingFace")
            return None

    def _try_trl_training(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset | None,
        max_steps: int | None,
    ) -> SkyRLTrainingResult | None:
        """Attempt to use TRL's GRPOTrainer for training.

        TRL (Transformers Reinforcement Learning) provides a well-tested
        implementation of GRPO that handles LoRA + gradient flow correctly.

        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            max_steps: Maximum training steps

        Returns:
            Training result or None if TRL is not available
        """
        try:
            from trl import GRPOTrainer, GRPOConfig
            from peft import LoraConfig
            from datasets import Dataset as HFDataset
            from transformers import AutoTokenizer, AutoModelForCausalLM

            logger.info("TRL library available - using GRPOTrainer")

            # Convert our dataset to HuggingFace format for TRL
            # TRL expects a dataset with 'prompt' column
            train_data = []
            self._trl_sample_metadata = {}  # Store metadata for reward function
            for i, sample in enumerate(train_dataset):
                prompt = sample.get("prompt", "")
                if isinstance(prompt, list):
                    # Convert chat format to string
                    prompt = self.tokenizer.apply_chat_template(
                        prompt, tokenize=False, add_generation_prompt=True
                    )
                train_data.append({"prompt": prompt})
                self._trl_sample_metadata[prompt] = {
                    "answer": sample.get("answer", ""),
                    "task_type": sample.get("task_type", "math"),
                    "sample_id": sample.get("sample_id", str(i)),
                }

            hf_train_dataset = HFDataset.from_list(train_data)

            hf_eval_dataset = None
            if eval_dataset:
                eval_data = []
                for i, sample in enumerate(eval_dataset):
                    prompt = sample.get("prompt", "")
                    if isinstance(prompt, list):
                        prompt = self.tokenizer.apply_chat_template(
                            prompt, tokenize=False, add_generation_prompt=True
                        )
                    eval_data.append({"prompt": prompt})
                    self._trl_sample_metadata[prompt] = {
                        "answer": sample.get("answer", ""),
                        "task_type": sample.get("task_type", "math"),
                        "sample_id": sample.get("sample_id", str(i)),
                    }
                hf_eval_dataset = HFDataset.from_list(eval_data)

            # Create TRL GRPOConfig
            trl_config = GRPOConfig(
                output_dir=self.config.output_dir,
                learning_rate=self.config.learning_rate,
                per_device_train_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                max_grad_norm=self.config.max_grad_norm,
                num_train_epochs=self.config.num_epochs,
                max_steps=max_steps if max_steps else -1,
                logging_steps=self.config.logging_steps,
                save_steps=self.config.save_steps,
                bf16=self.config.bf16,
                # GRPO-specific settings
                num_generations=self.config.group_size,
                max_prompt_length=self.config.max_prompt_length,
                max_completion_length=self.config.max_response_length,
                temperature=self.config.temperature,
                # Disable vLLM if not using it
                use_vllm=self.config.use_vllm,
            )

            # Create LoRA config if enabled
            peft_config = None
            if self.config.use_lora:
                peft_config = LoraConfig(
                    r=self.config.lora_r,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    target_modules=[
                        "q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                    ],
                    task_type="CAUSAL_LM",
                    bias="none",
                )

            # Create TRL trainer
            # TRL handles model loading, tokenizer, and LoRA internally
            trainer = GRPOTrainer(
                model=self.config.model_name,
                reward_funcs=self._trl_reward_function,
                args=trl_config,
                train_dataset=hf_train_dataset,
                eval_dataset=hf_eval_dataset,
                peft_config=peft_config,
            )

            # Train
            start_time = time.time()
            train_result = trainer.train()
            wall_time = time.time() - start_time

            # Save final model
            final_path = Path(self.config.output_dir) / "final"
            trainer.save_model(str(final_path))

            return SkyRLTrainingResult(
                model_path=final_path,
                final_loss=train_result.training_loss if hasattr(train_result, "training_loss") else 0.0,
                training_steps=train_result.global_step if hasattr(train_result, "global_step") else 0,
                wall_time_seconds=wall_time,
                metrics=train_result.metrics if hasattr(train_result, "metrics") else {},
                config=self.config.to_dict(),
            )

        except ImportError as e:
            logger.info(f"TRL not available ({e}), using fallback implementation")
            return None
        except Exception as e:
            logger.warning(f"TRL training failed ({e}), using fallback")
            import traceback
            traceback.print_exc()
            # Clean up GPU memory before fallback
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            return None

    def _trl_reward_function(
        self,
        prompts: list[str],
        completions: list[str],
        **kwargs,
    ) -> list[float]:
        """Reward function for TRL's GRPOTrainer.

        TRL passes prompts and completions separately.

        Args:
            prompts: List of prompts
            completions: List of model completions

        Returns:
            List of reward values
        """
        # Build samples from stored metadata
        samples = []
        for prompt in prompts:
            meta = self._trl_sample_metadata.get(prompt, {})
            samples.append({
                "answer": meta.get("answer", ""),
                "task_type": meta.get("task_type", "math"),
                "sample_id": meta.get("sample_id", ""),
            })

        # Call our reward function
        return self.reward_fn(completions, samples)

    def _wrap_reward_for_skyrl(
        self,
        responses: list[str],
        metadata_list: list[dict[str, Any]],
    ) -> list[float]:
        """Wrap reward function for SkyRL interface.

        SkyRL passes metadata dict instead of sample dict.

        Args:
            responses: List of model responses
            metadata_list: List of metadata dicts from PromptDataset

        Returns:
            List of reward values
        """
        # Convert metadata to sample format expected by our reward functions
        samples = [
            {
                "answer": meta.get("answer", ""),
                "task_type": meta.get("task_type", "math"),
                "sample_id": meta.get("sample_id", ""),
            }
            for meta in metadata_list
        ]
        return self.reward_fn(responses, samples)

    def _fallback_grpo_training(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset | None,
        max_steps: int | None,
    ) -> SkyRLTrainingResult:
        """Fallback GRPO training using optimized batched generation.

        This implements GRPO with async batched generation, similar to
        SkyRL's approach but without the full framework.

        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            max_steps: Maximum training steps

        Returns:
            Training result
        """
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR

        logger.info("Running optimized fallback GRPO training with batched generation")

        # Ensure model and tokenizer are loaded
        _ = self.model
        _ = self.tokenizer

        # Try to initialize vLLM for faster generation
        vllm_model = self._init_vllm()

        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Setup optimizer
        optimizer = AdamW(
            self._model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Convert dataset to list
        train_data = list(train_dataset)
        num_samples = len(train_data)

        # Calculate total steps
        total_steps = (
            max_steps if max_steps
            else num_samples // self.config.batch_size * self.config.num_epochs
        )

        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

        # Training loop
        start_time = time.time()
        global_step = 0
        total_loss = 0.0

        self._model.train()

        logger.info(f"Starting GRPO training: {num_samples} samples, {total_steps} steps")
        logger.info(f"Using {'vLLM' if vllm_model else 'HuggingFace'} for generation")

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            epoch_rewards: list[float] = []

            for batch_idx in range(0, num_samples, self.config.batch_size):
                if max_steps and global_step >= max_steps:
                    break

                batch_end = min(batch_idx + self.config.batch_size, num_samples)
                batch = train_data[batch_idx:batch_end]

                # Generate responses with batched generation
                if vllm_model:
                    batch_loss, batch_rewards = self._grpo_step_vllm(
                        batch, vllm_model, optimizer
                    )
                else:
                    batch_loss, batch_rewards = self._grpo_step_batched(
                        batch, optimizer
                    )

                epoch_loss += batch_loss
                epoch_rewards.extend(batch_rewards)
                total_loss += batch_loss

                global_step += 1
                scheduler.step()

                # Logging
                if global_step % self.config.logging_steps == 0:
                    avg_loss = epoch_loss / (batch_idx // self.config.batch_size + 1)
                    avg_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0
                    logger.info(
                        f"Step {global_step}: loss={avg_loss:.4f}, "
                        f"reward={avg_reward:.4f}, lr={scheduler.get_last_lr()[0]:.2e}"
                    )

                    self._metrics_history["loss"].append(avg_loss)
                    self._metrics_history["reward_mean"].append(avg_reward)

                # Save checkpoint
                if global_step % self.config.save_steps == 0:
                    checkpoint_path = output_dir / f"checkpoint-{global_step}"
                    self._save_checkpoint(checkpoint_path)

            if max_steps and global_step >= max_steps:
                break

        wall_time = time.time() - start_time

        # Save final model
        final_model_path = output_dir / "final"
        self._save_checkpoint(final_model_path)

        avg_loss = total_loss / max(global_step, 1)

        return SkyRLTrainingResult(
            model_path=final_model_path,
            final_loss=avg_loss,
            training_steps=global_step,
            wall_time_seconds=wall_time,
            metrics=self._metrics_history,
            config=self.config.to_dict(),
        )

    def _grpo_step_vllm(
        self,
        batch: list[dict[str, Any]],
        vllm_model: Any,
        optimizer: Any,
    ) -> tuple[float, list[float]]:
        """Execute GRPO step using vLLM for batched generation.

        Args:
            batch: Batch of training samples
            vllm_model: vLLM LLM instance
            optimizer: Optimizer instance

        Returns:
            Tuple of (loss, rewards)
        """
        from vllm import SamplingParams

        all_rewards: list[float] = []
        total_loss = 0.0

        # Prepare prompts for batched generation
        prompts = []
        for sample in batch:
            prompt = sample.get("prompt", "")
            if isinstance(prompt, list):
                prompt = self._tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            prompts.append(prompt)

        # Batched generation for all prompts * group_size
        expanded_prompts = prompts * self.config.group_size
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_response_length,
        )

        logger.info(f"Generating {len(expanded_prompts)} responses with vLLM...")
        outputs = vllm_model.generate(expanded_prompts, sampling_params)
        logger.info(f"Generated {len(outputs)} responses")

        # Reorganize outputs by prompt
        responses_by_prompt: list[list[str]] = [[] for _ in range(len(batch))]
        for i, output in enumerate(outputs):
            prompt_idx = i % len(batch)
            responses_by_prompt[prompt_idx].append(output.outputs[0].text)

        # Compute rewards and losses for each prompt's group
        for sample_idx, (sample, group_responses) in enumerate(zip(batch, responses_by_prompt)):
            # Compute rewards
            group_rewards = self.reward_fn(
                group_responses,
                [sample] * len(group_responses),
            )
            all_rewards.extend(group_rewards)

            # Compute GRPO loss using the model (requires gradients)
            batch_loss = self._compute_grpo_loss(
                sample, group_responses, group_rewards, optimizer
            )
            total_loss += batch_loss

        return total_loss / len(batch), all_rewards

    def _grpo_step_batched(
        self,
        batch: list[dict[str, Any]],
        optimizer: Any,
    ) -> tuple[float, list[float]]:
        """Execute GRPO step with batched HuggingFace generation.

        This is faster than the serial Verl fallback by generating
        all responses for a prompt in a single batch.

        Args:
            batch: Batch of training samples
            optimizer: Optimizer instance

        Returns:
            Tuple of (loss, rewards)
        """
        all_rewards: list[float] = []
        total_loss = 0.0

        for sample_idx, sample in enumerate(batch):
            logger.info(f"Processing sample {sample_idx + 1}/{len(batch)}")

            # Get prompt
            prompt = sample.get("prompt", "")
            if isinstance(prompt, list):
                prompt = self._tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                )

            # Tokenize and generate batch of responses
            inputs = self._tokenizer(
                [prompt] * self.config.group_size,  # Batch for parallel generation
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_prompt_length,
                padding=True,
            )

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Generate all group responses in one batch
            # Switch to eval mode for generation (important for gradient checkpointing)
            self._model.eval()

            # Debug: Check for NaN in model weights before generation
            for name, param in self._model.named_parameters():
                if param.requires_grad:
                    if torch.isnan(param).any():
                        logger.error(f"NaN in weights before generation: {name}")
                    if torch.isinf(param).any():
                        logger.error(f"Inf in weights before generation: {name}")

            with torch.no_grad():
                logger.info(f"  Generating {self.config.group_size} responses in batch...")
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_response_length,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self._tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                logger.info(f"  Generated batch of {outputs.sequences.shape[0]} responses")
            # Switch back to training mode for loss computation
            self._model.train()

            # Decode all responses
            group_responses = []
            group_response_ids = []
            group_old_log_probs = []

            prompt_length = inputs["input_ids"].shape[1]

            for i in range(outputs.sequences.shape[0]):
                response_ids = outputs.sequences[i][prompt_length:]
                response = self._tokenizer.decode(response_ids, skip_special_tokens=True)
                group_responses.append(response)
                group_response_ids.append(response_ids)

                # Compute old policy log prob
                if hasattr(outputs, "scores") and outputs.scores:
                    with torch.no_grad():
                        log_probs = torch.stack([
                            torch.log_softmax(score[i], dim=-1) for score in outputs.scores
                        ])
                        response_len = min(len(response_ids), len(outputs.scores))
                        selected_log_probs = log_probs[:response_len].gather(
                            -1, response_ids[:response_len].unsqueeze(-1)
                        ).squeeze(-1).sum()
                        group_old_log_probs.append(selected_log_probs.item())

            # Compute rewards
            group_rewards = self.reward_fn(
                group_responses,
                [sample] * len(group_responses),
            )
            all_rewards.extend(group_rewards)

            # Compute GRPO loss
            batch_loss = self._compute_grpo_loss_from_ids(
                inputs["input_ids"][0],
                group_response_ids,
                group_old_log_probs,
                group_rewards,
                optimizer,
            )
            total_loss += batch_loss

        return total_loss / len(batch), all_rewards

    def _compute_grpo_loss(
        self,
        sample: dict[str, Any],
        group_responses: list[str],
        group_rewards: list[float],
        optimizer: Any,
    ) -> float:
        """Compute GRPO loss and update model.

        Args:
            sample: Training sample
            group_responses: List of generated responses
            group_rewards: List of rewards for each response
            optimizer: Optimizer instance

        Returns:
            Loss value
        """
        # Get prompt
        prompt = sample.get("prompt", "")
        if isinstance(prompt, list):
            prompt = self._tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )

        # Tokenize prompt
        prompt_inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_prompt_length,
        )

        if torch.cuda.is_available():
            prompt_inputs = {k: v.cuda() for k, v in prompt_inputs.items()}

        prompt_ids = prompt_inputs["input_ids"][0]

        # Tokenize responses
        group_response_ids = []
        for response in group_responses:
            response_ids = self._tokenizer(
                response,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_response_length,
                add_special_tokens=False,
            )["input_ids"][0]
            if torch.cuda.is_available():
                response_ids = response_ids.cuda()
            group_response_ids.append(response_ids)

        # Compute loss using forward pass
        return self._compute_grpo_loss_from_ids(
            prompt_ids,
            group_response_ids,
            [],  # No cached log probs
            group_rewards,
            optimizer,
        )

    def _compute_grpo_loss_from_ids(
        self,
        prompt_ids: torch.Tensor,
        group_response_ids: list[torch.Tensor],
        group_old_log_probs: list[float],
        group_rewards: list[float],
        optimizer: Any,
    ) -> float:
        """Compute GRPO loss from tokenized inputs.

        Args:
            prompt_ids: Prompt token IDs
            group_response_ids: List of response token IDs
            group_old_log_probs: List of old policy log probs (empty if not cached)
            group_rewards: List of rewards
            optimizer: Optimizer instance

        Returns:
            Loss value
        """
        if not group_response_ids:
            return 0.0

        # Compute GRPO advantage (reward - mean)
        mean_reward = sum(group_rewards) / len(group_rewards)
        advantages = [r - mean_reward for r in group_rewards]

        # Accumulate losses from each response
        losses: list[torch.Tensor] = []

        for i, (response_ids, advantage) in enumerate(zip(group_response_ids, advantages)):
            if advantage == 0:
                continue

            # Create full sequence
            full_ids = torch.cat([prompt_ids, response_ids]).unsqueeze(0)

            # Forward pass with gradient tracking enabled
            # Required for proper backprop through quantized models with LoRA
            with torch.enable_grad():
                outputs = self._model(full_ids, use_cache=False)
                logits = outputs.logits

                # Check for NaN in logits (debug)
                if torch.isnan(logits).any():
                    logger.warning(f"NaN detected in logits during loss computation")
                if torch.isinf(logits).any():
                    logger.warning(f"Inf detected in logits during loss computation")

            # Compute log probs for response tokens
            prompt_len = len(prompt_ids)
            response_logits = logits[0, prompt_len-1:-1, :]
            response_log_probs = torch.log_softmax(response_logits, dim=-1)

            # Select log probs for actual tokens
            response_len = min(len(response_ids), response_logits.shape[0])
            new_log_prob = response_log_probs[:response_len].gather(
                -1, response_ids[:response_len].unsqueeze(-1)
            ).squeeze(-1).sum()

            # Compute policy ratio if old log probs available
            if group_old_log_probs and i < len(group_old_log_probs):
                old_log_prob = group_old_log_probs[i]
                log_ratio = new_log_prob - old_log_prob
                ratio = torch.exp(log_ratio)

                # Clipped objective
                clipped_ratio = torch.clamp(
                    ratio,
                    1 - self.config.clip_ratio,
                    1 + self.config.clip_ratio,
                )

                surrogate1 = ratio * advantage
                surrogate2 = clipped_ratio * advantage
                # Negate because we minimize loss (maximize reward)
                losses.append(-torch.min(surrogate1, surrogate2))
            else:
                # Simple policy gradient without ratio
                losses.append(-new_log_prob * advantage)

        # Average and backprop
        if losses:
            # Stack and mean to get proper gradient flow
            batch_loss = torch.stack(losses).mean()
            loss_value = batch_loss.item()

            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self._model.parameters(),
                self.config.max_grad_norm,
            )
            optimizer.step()

            return loss_value

        return 0.0

    def _save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        path.mkdir(parents=True, exist_ok=True)

        if self._model is not None:
            self._model.save_pretrained(path)
        if self._tokenizer is not None:
            self._tokenizer.save_pretrained(path)

        logger.info(f"Checkpoint saved to {path}")

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset | None = None,
        max_steps: int | None = None,
        resume_from_checkpoint: str | Path | None = None,
    ) -> SkyRLTrainingResult:
        """Train the model using GRPO.

        Args:
            train_dataset: Training dataset with 'prompt', 'answer', 'task_type'
            eval_dataset: Optional evaluation dataset
            max_steps: Maximum training steps (overrides config)
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            SkyRLTrainingResult with model path and metrics
        """
        # Only load tokenizer first - TRL will load its own model
        # This prevents double model loading and OOM issues
        _ = self.tokenizer

        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
            self.load_model(resume_from_checkpoint)

        # Try TRL first (well-tested GRPO) - it handles model loading internally
        # Only fall back to custom implementation if TRL is not available
        result = self._try_trl_training(train_dataset, eval_dataset, max_steps)

        if result is None:
            # TRL not available or failed - load our own model for fallback
            _ = self.model
            result = self._fallback_grpo_training(
                train_dataset, eval_dataset, max_steps
            )

        # Save result
        output_dir = Path(self.config.output_dir)
        result.save(output_dir)

        logger.info(
            f"Training complete: {result.training_steps} steps, "
            f"loss={result.final_loss:.4f}, time={result.wall_time_seconds:.1f}s"
        )

        return result

    def load_model(self, path: Path | str) -> None:
        """Load a trained model from disk.

        Args:
            path: Path to load the model from
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = Path(path)

        logger.info(f"Loading model from {path}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=True,
        )

        dtype = torch.bfloat16 if self.config.bf16 and torch.cuda.is_available() else torch.float32

        self._model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            torch_dtype=dtype,
        )

    def save_model(self, path: Path | str) -> None:
        """Save the trained model.

        Args:
            path: Path to save the model
        """
        self._save_checkpoint(Path(path))


def train_skyrl_grpo(
    config_path: Path | str | None = None,
    math_samples: int = 10000,
    logic_samples: int = 5000,
    code_samples: int = 2000,
    max_steps: int | None = None,
    output_dir: Path | str | None = None,
    wandb_enabled: bool = False,
) -> SkyRLTrainingResult:
    """Convenience function to run SkyRL GRPO training.

    Args:
        config_path: Path to config YAML file
        math_samples: Number of math training samples
        logic_samples: Number of logic training samples
        code_samples: Number of code training samples
        max_steps: Maximum training steps
        output_dir: Output directory (overrides config)
        wandb_enabled: Whether to enable WandB logging

    Returns:
        SkyRLTrainingResult
    """
    from src.rl.skyrl_data_pipeline import load_skyrl_datasets

    # Load config
    if config_path:
        config = SkyRLConfig.from_yaml(config_path)
    else:
        config = SkyRLConfig()

    # Override settings
    if output_dir:
        config.output_dir = str(output_dir)

    # Initialize WandB if enabled
    if wandb_enabled:
        import wandb
        wandb.init(
            project="lore-skyrl-grpo",
            config=config.to_dict(),
        )

    # Create trainer
    trainer = LORESkyRLTrainer(config=config)

    # Load datasets
    logger.info("Loading datasets")
    datasets = load_skyrl_datasets(
        math_samples=math_samples,
        logic_samples=logic_samples,
        code_samples=code_samples,
        tokenizer=trainer.tokenizer,
    )

    # Train
    result = trainer.train(
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        max_steps=max_steps,
    )

    # Finish WandB
    if wandb_enabled:
        import wandb
        wandb.finish()

    return result
