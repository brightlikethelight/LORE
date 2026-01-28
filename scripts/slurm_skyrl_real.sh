#!/bin/bash
#SBATCH --job-name=skyrl_grpo
#SBATCH --output=logs/skyrl_real_%j.out
#SBATCH --error=logs/skyrl_real_%j.err
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=main

# SkyRL GRPO Training with actual SkyRL library
# Requires Python 3.12 conda environment with SkyRL installed

set -e
mkdir -p logs

# Load conda and activate Python 3.12 environment
source ~/.bashrc
module load miniconda/3
conda activate lore312

cd /home/mila/t/thomas.jiralerspong/LORE

echo "=============================================="
echo "SkyRL GRPO Training (Real SkyRL)"
echo "=============================================="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "SLURM Job ID: ${SLURM_JOB_ID:-local}"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || echo "No GPU"
echo ""

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-~/.cache/huggingface}
export RAY_DEDUP_LOGS=0

# Set CUDA_HOME for flash_attn compilation
export CUDA_HOME=/cvmfs/ai.mila.quebec/apps/x86_64/common/cuda/12.1.1
export PATH=$CUDA_HOME/bin:$PATH

# Install flash_attn if not already installed
python -c "import flash_attn" 2>/dev/null || {
    echo "Installing flash_attn on GPU node..."
    pip install flash-attn --no-build-isolation --no-cache-dir 2>&1 | tail -5
}

# Data directory
DATA_DIR="${DATA_DIR:-data/skyrl_gsm8k}"

# Model
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

echo "Data dir: ${DATA_DIR}"
echo "Model: ${MODEL}"
echo "=============================================="

# Run SkyRL GRPO training
# Single GPU configuration with colocated inference
python -m skyrl_train.entrypoints.main_base \
  data.train_data="['${DATA_DIR}/train.parquet']" \
  data.val_data="['${DATA_DIR}/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="${MODEL}" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=1 \
  trainer.placement.critic_num_gpus_per_node=1 \
  trainer.placement.ref_num_gpus_per_node=1 \
  generator.num_inference_engines=1 \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=3 \
  trainer.eval_batch_size=32 \
  trainer.eval_before_train=true \
  trainer.eval_interval=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=32 \
  trainer.policy_mini_batch_size=16 \
  trainer.micro_forward_batch_size_per_gpu=8 \
  trainer.micro_train_batch_size_per_gpu=8 \
  trainer.ckpt_interval=1 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=256 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=false \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=4 \
  generator.gpu_memory_utilization=0.7 \
  trainer.logger="console" \
  trainer.project_name="lore-skyrl" \
  trainer.run_name="skyrl_smoke_${SLURM_JOB_ID:-local}" \
  trainer.resume_mode=null \
  trainer.ckpt_path="results/skyrl_real_${SLURM_JOB_ID:-local}"

echo ""
echo "=============================================="
if [ $? -eq 0 ]; then
    echo "SkyRL training PASSED!"
else
    echo "SkyRL training FAILED!"
fi
echo "=============================================="
