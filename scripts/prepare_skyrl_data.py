#!/usr/bin/env python3
"""Prepare data for SkyRL GRPO training.

Converts our multi-task datasets to SkyRL's expected parquet format.
"""

import argparse
import json
import re
from pathlib import Path

import polars as pl
from datasets import load_dataset


def extract_solution(solution_str: str) -> str:
    """Extract the final answer from GSM8K solution string."""
    solution = re.search(r"#### (\-?[0-9\.\,]+)", solution_str)
    if solution is None:
        return ""
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


def prepare_gsm8k_data(output_dir: Path, num_samples: int = 1000):
    """Prepare GSM8K data in SkyRL format."""
    print("Loading GSM8K dataset...")
    ds = load_dataset("openai/gsm8k", "main", split="train")

    instruction = 'Let\'s think step by step and output the final answer after "####".'

    data = []
    for i, sample in enumerate(ds):
        if i >= num_samples:
            break

        question = sample["question"] + " " + instruction
        answer_raw = sample["answer"]
        solution = extract_solution(answer_raw)

        # SkyRL format with env_class and reward_spec
        prompt = [
            {"role": "user", "content": question}
        ]

        data.append({
            "data_source": "openai/gsm8k",
            "prompt": prompt,  # SkyRL handles serialization
            "env_class": "gsm8k",
            "reward_spec": {
                "method": "rule",
                "ground_truth": solution,
            },
            "extra_info": {
                "split": "train",
                "index": i,
                "answer": answer_raw,
                "question": sample["question"],
            },
        })

    df = pl.DataFrame(data)

    # Split into train/val
    train_size = int(len(data) * 0.9)
    train_df = df[:train_size]
    val_df = df[train_size:]

    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.write_parquet(output_dir / "train.parquet")
    val_df.write_parquet(output_dir / "validation.parquet")

    print(f"Saved {len(train_df)} train samples to {output_dir / 'train.parquet'}")
    print(f"Saved {len(val_df)} val samples to {output_dir / 'validation.parquet'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/skyrl_gsm8k"))
    parser.add_argument("--num-samples", type=int, default=1000)
    args = parser.parse_args()

    prepare_gsm8k_data(args.output_dir, args.num_samples)


if __name__ == "__main__":
    main()
