#!/usr/bin/env bash
# Multi-turn GRPO training for ScienceWorld using sglang
#
# This script trains a model using multi-turn interaction with ScienceWorld environment.
# The model generates one action at a time, receives observation, and continues.
#
# Usage:
#   conda activate verl
#   CUDA_VISIBLE_DEVICES=1 bash examples/scienceworld_grpo/run_multiturn_training.sh
#
# Requirements:
#   pip install "sglang[all]==0.4.6.post5"
#   export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True

set -xeuo pipefail

########################### user-adjustable ###########################
# Model configuration
MODEL_PATH=${MODEL_PATH:-/workspace/models/Qwen3-1.7B}

# Hardware configuration
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}

# Training hyperparameters (aligned with TCOD paper)
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-10240}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-6144}

# Learning rate and optimization (same as TCOD paper)
ACTOR_LR=${ACTOR_LR:-1e-6}
GRAD_CLIP=${GRAD_CLIP:-1.0}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}

# Rollout configuration
ROLLOUT_N=${ROLLOUT_N:-4}
ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.7}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}

# Training schedule (250 steps as in TCOD paper)
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
SAVE_FREQ=${SAVE_FREQ:-250}
TEST_FREQ=${TEST_FREQ:-5}
SEED=${SEED:-42}

# Data configuration (all 30 tasks)
DATA_DIR=${DATA_DIR:-/workspace/data/scienceworld_all}

# Multi-turn configuration
MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-30}

# Experiment tracking
PROJECT_NAME=${PROJECT_NAME:-verl_grpo_scienceworld_multiturn}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-scienceworld_multiturn_grpo_$(date +%Y%m%d_%H%M)}
########################### end user-adjustable ###########################

# Get project directory
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/scienceworld_grpo/config"

# Set sglang environment variable for single machine
export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True

echo "=========================================="
echo "ScienceWorld Multi-turn GRPO Training"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "GPUs: ${NGPUS_PER_NODE}"
echo "Batch size: ${TRAIN_BATCH_SIZE}"
echo "Rollout N: ${ROLLOUT_N}"
echo "Data dir: ${DATA_DIR}"
echo "Max assistant turns: ${MAX_ASSISTANT_TURNS}"
echo "NOTE: Using REAL ScienceWorld environment (all 30 tasks, multi-turn)"
echo "=========================================="

# Check if data exists
if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    echo "Error: Training data not found at ${DATA_DIR}/train.parquet"
    echo "Please run data generation first:"
    echo "  bash examples/scienceworld_grpo/generate_all_data.sh"
    exit 1
fi

# Launch training
python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='scienceworld_multiturn_grpo' \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/val.parquet \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.actor.optim.grad_clip=${GRAD_CLIP} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE} \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_ASSISTANT_TURNS} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.seed=${SEED} \
    "$@"

echo ""
echo "Training complete!"
echo "Checkpoint saved to: checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}/"
