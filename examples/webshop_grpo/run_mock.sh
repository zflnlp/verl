#!/usr/bin/env bash
# GRPO | WebShop Mock Training | v0.4.1 Compatible
#
# This script trains a model using MOCK WebShop environment.
# It does NOT require a real WebShop server - perfect for testing.
#
# Usage:
#   bash examples/webshop_grpo/run_mock.sh
#
# For real WebShop training, use:
#   bash examples/webshop_grpo/run_real.sh

set -xeuo pipefail

########################### user-adjustable ###########################
# Model configuration
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}

# Hardware configuration
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}

# Training hyperparameters
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}

# Learning rate
ACTOR_LR=${ACTOR_LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}

# Rollout configuration
ROLLOUT_N=${ROLLOUT_N:-4}
ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.6}

# Training schedule
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
SAVE_FREQ=${SAVE_FREQ:-5}
TEST_FREQ=${TEST_FREQ:-2}

# Data configuration
DATA_DIR=${DATA_DIR:-$HOME/data/webshop}

# Experiment tracking
PROJECT_NAME=${PROJECT_NAME:-verl_grpo_webshop_mock}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-webshop_mock_grpo_$(date +%Y%m%d_%H%M)}
########################### end user-adjustable ###########################

# Generate data if not exists
if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    echo "Generating mock training data..."
    python3 examples/webshop_grpo/data_preprocess.py \
        --local_save_dir ${DATA_DIR} \
        --num_tasks 100 \
        --seed 42
fi

# Get project directory
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/webshop_grpo/config"

echo "=========================================="
echo "WebShop GRPO Training (MOCK Mode)"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "GPUs: ${NGPUS_PER_NODE}"
echo "Batch size: ${TRAIN_BATCH_SIZE}"
echo "Rollout N: ${ROLLOUT_N}"
echo "Data dir: ${DATA_DIR}"
echo "NOTE: Using MOCK WebShop environment"
echo "=========================================="

# Launch training
python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='webshop_interaction_config' \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=15 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${CONFIG_PATH}/webshop_tool_config.yaml" \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="${CONFIG_PATH}/webshop_interaction_config.yaml" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.val_before_train=False \
    "$@"
