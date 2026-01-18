#!/usr/bin/env python3
"""Run GEPA evolution with improved evaluation (v3)."""

import os
os.environ['HF_HOME'] = '/home/mila/t/thomas.jiralerspong/.cache/huggingface'

import asyncio
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
log_file = Path("/home/mila/t/thomas.jiralerspong/LORE/logs/evolution_v3.log")
log_file.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def main():
    logger.info(f"Starting GEPA Evolution v3 (improved evaluation) at {datetime.now()}")
    
    from src.data.ultrainteract import UltraInteractLoader, create_gepa_dataset
    from src.optimization.ultrainteract_gepa import (
        run_ultrainteract_evolution,
        DEFAULT_SEED_PROMPT,
    )
    
    # Config
    task_lm = "anthropic/claude-opus-4-5-20251101"
    train_size = 100
    val_size = 30
    max_metric_calls = 200
    
    logger.info(f"Config: {task_lm}, train={train_size}, val={val_size}")
    
    # Load data
    logger.info("Loading UltraInteract dataset...")
    loader = UltraInteractLoader(seed=42)
    splits = loader.load_stratified(train_size=train_size, val_size=val_size, test_size=0)
    
    train_dist = loader.get_task_distribution(splits["train"])
    val_dist = loader.get_task_distribution(splits["val"])
    logger.info(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}")
    logger.info(f"Train dist: {train_dist}")
    logger.info(f"Val dist: {val_dist}")
    
    # Check sample answers
    val_data = create_gepa_dataset(splits['val'])
    logger.info("Sample expected answers:")
    for sample in val_data[:5]:
        task_type = sample['additional_context']['task_type']
        answer = sample['answer'][:50] if sample['answer'] else 'None'
        logger.info(f"  [{task_type}]: {answer}...")
    
    # Run evolution
    output_dir = Path("/home/mila/t/thomas.jiralerspong/LORE/results/ui_evolution_v3")
    logger.info(f"Starting GEPA evolution...")
    
    result = await run_ultrainteract_evolution(
        trainset=splits["train"],
        valset=splits["val"],
        task_lm=task_lm,
        reflection_lm=task_lm,
        max_metric_calls=max_metric_calls,
        seed_prompt=DEFAULT_SEED_PROMPT,
        output_dir=output_dir,
        use_wandb=False,
        seed=42,
    )
    
    # Log results
    logger.info("="*60)
    logger.info("EVOLUTION COMPLETE")
    logger.info("="*60)
    logger.info(f"Generations: {result.generations_completed}")
    logger.info(f"Total evaluations: {result.total_evaluations}")
    logger.info(f"Wall time: {result.wall_time_seconds:.1f}s")
    logger.info(f"\nFinal fitness:")
    for task, score in result.fitness.items():
        logger.info(f"  {task}: {score:.1%}")
    logger.info(f"\nBest prompt (first 500 chars):")
    logger.info(result.best_general_prompt[:500])
    logger.info(f"\nResults saved to: {output_dir}")
    
    return result

if __name__ == "__main__":
    asyncio.run(main())
