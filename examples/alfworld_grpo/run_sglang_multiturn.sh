#!/usr/bin/env bash
# GRPO | ALFWorld Multi-Turn Training with SGLang | v0.4.1 Compatible
#
# This script trains a model using SGLang for multi-turn rollout.
# SGLang supports true multi-turn interaction with tool calling.
#
# Usage:
#   bash examples/alfworld_grpo/run_sglang_multiturn.sh
#
# For vLLM single-turn training, use:
#   bash examples/alfworld_grpo/run_mock.sh

set -xeuo pipefail

########################### user-adjustable ###########################
# Model configuration
MODEL_PATH=${MODEL_PATH:-/workspace/models/Qwen3-1.7B}

# Hardware configuration
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}

# Training hyperparameters
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-10240}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}

# Learning rate
ACTOR_LR=${ACTOR_LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-1.0}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}

# Rollout configuration
ROLLOUT_N=${ROLLOUT_N:-4}
ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.7}

# Training schedule
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
SAVE_FREQ=${SAVE_FREQ:-5}
TEST_FREQ=${TEST_FREQ:-2}

# Data configuration
DATA_DIR=${DATA_DIR:-/workspace/data/alfworld}

# Experiment tracking
PROJECT_NAME=${PROJECT_NAME:-verl_grpo_alfworld_sglang}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-alfworld_sglang_multiturn_$(date +%Y%m%d_%H%M)}
########################### end user-adjustable ###########################

# Get project directory
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/alfworld_grpo/config"

echo "=========================================="
echo "ALFWorld GRPO Multi-Turn Training (SGLang)"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "GPUs: ${NGPUS_PER_NODE}"
echo "Batch size: ${TRAIN_BATCH_SIZE}"
echo "Rollout N: ${ROLLOUT_N}"
echo "Data dir: ${DATA_DIR}"
echo "NOTE: Using SGLang for multi-turn rollout"
echo "=========================================="

# Launch training
python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='alfworld_sglang_multiturn' \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${CONFIG_PATH}/alfworld_tool_config.yaml" \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="${CONFIG_PATH}/alfworld_interaction_config.yaml" \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    "$@"
