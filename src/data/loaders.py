"""Dataset loading utilities for LORE experiments."""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import yaml
from datasets import load_dataset as hf_load_dataset


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""

    name: str
    source: str
    split: str
    subset: str | list[str] | None = None
    sample_size: int | None = None
    seed: int = 42
    columns: dict[str, str] = field(default_factory=dict)
    preprocessing: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, config: dict[str, Any]) -> "DatasetConfig":
        """Create from dictionary."""
        return cls(
            name=name,
            source=config["source"],
            split=config["split"],
            subset=config.get("subset"),
            sample_size=config.get("sample_size"),
            seed=config.get("seed", 42),
            columns=config.get("columns", {}),
            preprocessing=config.get("preprocessing", {}),
        )


@dataclass
class DataPoint:
    """A single data point for evaluation."""

    id: str
    question: str
    answer: str | None = None
    choices: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "choices": self.choices,
            "metadata": self.metadata,
        }


class DataLoader:
    """Loader for experiment datasets."""

    def __init__(
        self,
        config_path: Path | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        """Initialize the loader.

        Args:
            config_path: Path to datasets.yaml config
            cache_dir: Directory for caching datasets
        """
        self.cache_dir = cache_dir or Path("./data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if config_path:
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}

    def _get_cache_path(self, config: DatasetConfig) -> Path:
        """Generate cache path for a dataset configuration."""
        # Create hash of config for cache key
        config_str = f"{config.source}_{config.split}_{config.subset}_{config.sample_size}_{config.seed}"
        cache_key = hashlib.md5(config_str.encode()).hexdigest()[:12]
        return self.cache_dir / f"{config.name}_{cache_key}.parquet"

    def _preprocess_answer(
        self,
        answer: str,
        preprocessing: dict[str, Any],
    ) -> str:
        """Apply preprocessing to extract/normalize answer."""
        if preprocessing.get("extract_final_answer", False):
            pattern = preprocessing.get("answer_pattern", r"#### (.*)")
            match = re.search(pattern, answer, re.DOTALL)
            if match:
                return match.group(1).strip()

        return answer.strip()

    def _format_mcq(
        self,
        question: str,
        choices: list[str],
    ) -> str:
        """Format multiple choice question with options."""
        options = "\n".join(
            f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)
        )
        return f"{question}\n\n{options}"

    def load(self, config: DatasetConfig) -> list[DataPoint]:
        """Load dataset according to configuration.

        Args:
            config: Dataset configuration

        Returns:
            List of DataPoint objects
        """
        # Check cache
        cache_path = self._get_cache_path(config)
        # For simplicity, not implementing full cache logic here

        # Load from HuggingFace
        load_kwargs: dict[str, Any] = {
            "path": config.source,
            "split": config.split,
        }

        if config.subset:
            if isinstance(config.subset, list):
                # Load multiple subsets and combine
                datasets = []
                for subset in config.subset:
                    load_kwargs["name"] = subset
                    datasets.append(hf_load_dataset(**load_kwargs))
                from datasets import concatenate_datasets
                dataset = concatenate_datasets(datasets)
            else:
                load_kwargs["name"] = config.subset
                dataset = hf_load_dataset(**load_kwargs)
        else:
            dataset = hf_load_dataset(**load_kwargs)

        # Sample if needed
        if config.sample_size and len(dataset) > config.sample_size:
            dataset = dataset.shuffle(seed=config.seed).select(range(config.sample_size))

        # Convert to DataPoints
        data_points = []
        question_col = config.columns.get("question", "question")
        answer_col = config.columns.get("answer", "answer")
        choices_col = config.columns.get("choices", "choices")

        for i, item in enumerate(dataset):
            # Get question
            question = item.get(question_col, "")

            # Handle MCQ formatting
            choices = None
            if choices_col in item and config.preprocessing.get("format_mcq", False):
                choices = item[choices_col]
                question = self._format_mcq(question, choices)

            # Get and preprocess answer
            answer = item.get(answer_col, "")
            if isinstance(answer, int) and choices:
                # Convert index to letter for MCQ
                answer = chr(65 + answer)
            elif isinstance(answer, str):
                answer = self._preprocess_answer(answer, config.preprocessing)

            # Generate ID
            item_id = f"{config.name}_{i}"

            data_points.append(
                DataPoint(
                    id=item_id,
                    question=question,
                    answer=answer,
                    choices=choices,
                    metadata={
                        "source": config.source,
                        "split": config.split,
                        "original_index": i,
                    },
                )
            )

        return data_points

    def load_by_name(self, dataset_name: str, category: str = "training") -> list[DataPoint]:
        """Load dataset by name from config.

        Args:
            dataset_name: Name of dataset in config
            category: Category (training, evaluation, safety)

        Returns:
            List of DataPoint objects
        """
        datasets_config = self.config.get("datasets", {})
        category_config = datasets_config.get(category, {})

        if dataset_name not in category_config:
            raise ValueError(f"Dataset '{dataset_name}' not found in {category}")

        config = DatasetConfig.from_dict(dataset_name, category_config[dataset_name])
        return self.load(config)

    def iter_batches(
        self,
        data_points: list[DataPoint],
        batch_size: int = 32,
    ) -> Iterator[list[DataPoint]]:
        """Iterate over data points in batches.

        Args:
            data_points: List of data points
            batch_size: Batch size

        Yields:
            Batches of data points
        """
        for i in range(0, len(data_points), batch_size):
            yield data_points[i : i + batch_size]


def load_dataset(
    source: str,
    split: str = "test",
    sample_size: int | None = None,
    seed: int = 42,
) -> list[DataPoint]:
    """Convenience function to load a dataset.

    Args:
        source: HuggingFace dataset path
        split: Dataset split
        sample_size: Optional sample size
        seed: Random seed for sampling

    Returns:
        List of DataPoint objects
    """
    config = DatasetConfig(
        name=source.split("/")[-1],
        source=source,
        split=split,
        sample_size=sample_size,
        seed=seed,
    )
    loader = DataLoader()
    return loader.load(config)


def get_eval_dataset(name: str, config_path: Path | None = None) -> list[DataPoint]:
    """Get evaluation dataset by name.

    Args:
        name: Dataset name (e.g., "gsm8k_test", "gpqa_diamond")
        config_path: Optional path to datasets.yaml

    Returns:
        List of DataPoint objects
    """
    config_path = config_path or Path("configs/datasets.yaml")
    loader = DataLoader(config_path=config_path)
    return loader.load_by_name(name, category="evaluation")
