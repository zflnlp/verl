#!/usr/bin/env bash
# Multi-turn GRPO training for ScienceWorld — verl main (latest)
#

# ScienceWorld needs high file descriptor limit for Java gateway
# Uses the new agent_loop architecture with custom ScienceWorldAgentLoop.
# Compatible with CUDA 13.1 on the new machine.

# Uses the new agent_loop architecture with custom ScienceWorldAgentLoop.
# Compatible with CUDA 13.1 on the new machine.
#
# Usage:
#   bash examples/scienceworld_grpo/run_multiturn_v2.sh
#
# For old machine (CUDA 12.4), use run_multiturn_training.sh instead.

set -xeuo pipefail

# Disable DeepGEMM (not available in this environment)
export VLLM_USE_DEEP_GEMM=0
export VLLM_SKIP_WARMUP=1
# Fix flashinfer version mismatch
export FLASHINFER_DISABLE_VERSION_CHECK=1

########################### config ###########################
MODEL_PATH=${MODEL_PATH:-/workspace/models/Qwen3-1.7B-SFT-v2}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-10240}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
ACTOR_LR=${ACTOR_LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
ROLLOUT_N=${ROLLOUT_N:-4}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.7}
SAVE_FREQ=${SAVE_FREQ:-250}
TEST_FREQ=${TEST_FREQ:-5}
SEED=${SEED:-42}
DATA_DIR=${DATA_DIR:-/workspace/data/scienceworld_all}
MAX_STEPS=${MAX_STEPS:-50}
PROJECT_NAME=${PROJECT_NAME:-verl_grpo_scienceworld}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-scienceworld_grpo_$(date +%Y%m%d_%H%M)}
###########################

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/scienceworld_grpo/config"

echo "=========================================="
echo "ScienceWorld GRPO Training (verl main)"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "GPUs: ${NGPUS_PER_NODE}"
echo "Batch: ${TRAIN_BATCH_SIZE}"
echo "Rollout N: ${ROLLOUT_N}"
echo "Agent: scienceworld_agent"
echo "=========================================="

[ -f "${DATA_DIR}/train.parquet" ] || {
    echo "Error: ${DATA_DIR}/train.parquet not found"
    exit 1
}

# Ensure our agent loop module is importable
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='scienceworld_multiturn_v2' \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/val.parquet \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_STEPS} \
    actor_rollout_ref.rollout.agent.default_agent_loop=scienceworld_agent \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.seed=${SEED} \
    "$@"
