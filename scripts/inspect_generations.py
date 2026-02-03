#!/usr/bin/env python3
"""Inspect generations from trained GRPO models vs baseline."""

import sys
sys.path.insert(0, '.')

import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from src.rl.data_pipeline import load_ultrainteract_for_grpo
from src.rl.reward import TaskRewardFunction


def load_model(base_model: str, adapter_path: str | None = None):
    """Load model with optional LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    """Generate a response."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response


def main():
    base_model = "Qwen/Qwen2.5-7B-Instruct"
    adapter_path = "results/sweep_high_lr_20260122_121415/final"

    # Check if adapter exists
    if not Path(adapter_path).exists():
        # Try another path
        adapter_path = "results/sweep_conservative_20260122_121415/final"

    print(f"Base model: {base_model}")
    print(f"Adapter: {adapter_path}")
    print("=" * 60)

    # Load tokenizer for data
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Load some test samples
    datasets = load_ultrainteract_for_grpo(
        train_size=100,
        val_size=50,
        tokenizer=tokenizer,
        seed=42,
        task_types=["Coding", "Math_CoT", "Math_PoT", "Logic"],
    )

    reward_fn = TaskRewardFunction()

    # Get 2 samples of each task type
    samples_by_task = {"Coding": [], "Math_CoT": [], "Math_PoT": [], "Logic": []}
    for sample in datasets["val"]:
        task_type = sample.get("task_type", "unknown")
        if task_type in samples_by_task and len(samples_by_task[task_type]) < 2:
            samples_by_task[task_type].append(sample)

    # Load baseline model
    print("\nLoading baseline model...")
    base_model_obj, base_tokenizer = load_model(base_model)

    # Load fine-tuned model
    print("Loading fine-tuned model...")
    ft_model, ft_tokenizer = load_model(base_model, adapter_path)

    # Compare generations
    for task_type, samples in samples_by_task.items():
        print(f"\n{'='*60}")
        print(f"TASK TYPE: {task_type}")
        print("=" * 60)

        for i, sample in enumerate(samples[:1]):  # Just 1 per task for brevity
            prompt = sample["prompt"]
            expected = sample.get("answer", "N/A")

            print(f"\n--- Sample {i+1} ---")
            print(f"PROMPT (truncated): {prompt[:300]}...")
            print(f"\nEXPECTED: {expected[:200]}...")

            # Generate baseline
            print(f"\nBASELINE RESPONSE:")
            base_response = generate_response(base_model_obj, base_tokenizer, prompt)
            print(base_response[:500])
            base_reward = reward_fn(base_response, sample)
            print(f"\n[Reward: {base_reward.reward}, Correct: {base_reward.is_correct}]")

            # Generate fine-tuned
            print(f"\nFINE-TUNED RESPONSE:")
            ft_response = generate_response(ft_model, ft_tokenizer, prompt)
            print(ft_response[:500])
            ft_reward = reward_fn(ft_response, sample)
            print(f"\n[Reward: {ft_reward.reward}, Correct: {ft_reward.is_correct}]")

            print("-" * 40)


if __name__ == "__main__":
    main()
