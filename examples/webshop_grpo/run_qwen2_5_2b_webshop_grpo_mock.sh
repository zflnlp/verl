#!/usr/bin/env bash
# GRPO | Qwen2.5-2B | Mock WebShop Environment | FSDP Training
#
# This script trains a Qwen2.5-2B model using MOCK WebShop tools.
# It does NOT require a real WebShop server - perfect for testing
# the training pipeline or running on machines without WebShop.
#
# Usage:
#   bash examples/webshop_grpo/run_qwen2_5_2b_webshop_grpo_mock.sh
#
# For real WebShop training, use:
#   bash examples/webshop_grpo/run_qwen2_5_2b_webshop_grpo.sh

set -xeuo pipefail

# Force vllm to use V0 engine for compatibility with older versions
export VLLM_USE_V1=0

########################### user-adjustable ###########################
# Model configuration
# For local models, use absolute path like: /root/.cache/modelscope/hub/models/Qwen/Qwen3-1.7B
# For HuggingFace models, use: Qwen/Qwen3-1.7B
MODEL_PATH=${MODEL_PATH:-/workspace/models/Qwen3-1.7B}

# Hardware configuration
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}

# Training hyperparameters (optimized for testing)
train_batch_size=${TRAIN_BATCH_SIZE:-32}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-8}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-2048}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}

# Learning rate
actor_lr=${ACTOR_LR:-1e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.001}
entropy_coeff=${ENTROPY_COEFF:-0.01}

# Rollout configuration
rollout_n=${ROLLOUT_N:-4}
rollout_tp=${ROLLOUT_TP:-1}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.5}

# Training schedule
total_epochs=${TOTAL_EPOCHS:-3}
save_freq=${SAVE_FREQ:-5}
test_freq=${TEST_FREQ:-2}

# Experiment tracking
PROJECT_NAME=${PROJECT_NAME:-verl_grpo_webshop_mock}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_1.7b_webshop_mock_grpo_$(date +%Y%m%d_%H%M)}
########################### end user-adjustable ###########################

########################### derived defaults ###########################
DEVICE=${DEVICE:-$(python3 -c 'import torch_npu' 2>/dev/null && echo npu || echo gpu)}

case "${DEVICE}" in
    gpu)
        actor_param_offload=False
        actor_optimizer_offload=False
        n_trainer_devices=${NGPUS_PER_NODE}
        ;;
    npu)
        NGPUS_PER_NODE=8
        n_trainer_devices=${NGPUS_PER_NODE}
        actor_param_offload=True
        actor_optimizer_offload=True
        ;;
esac

########################### data paths ###########################
# Data directory - change this to your actual data path
DATA_DIR=${DATA_DIR:-/workspace/data/webshop_test}

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="['${DATA_DIR}/train.parquet']"
    data.val_files="['${DATA_DIR}/test.parquet']"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_param_offload}
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_optimizer_offload}
    actor_rollout_ref.actor.ppo_epochs=1
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    # Multi-turn with MOCK tools
    actor_rollout_ref.rollout.multi_turn.enable=True
    actor_rollout_ref.rollout.multi_turn.function_tool_path=examples/webshop_grpo/mock_webshop_tools.py
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=15
    actor_rollout_ref.rollout.multi_turn.max_user_turns=1
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024
    actor_rollout_ref.rollout.multi_turn.format=hermes
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
    actor_rollout_ref.rollout.agent.num_workers=4
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

REWARD=(
    reward.custom_reward_function.path=examples/webshop_grpo/reward_function.py
    reward.custom_reward_function.name=compute_webshop_reward
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${n_trainer_devices}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
    trainer.val_before_train=True
    trainer.val_only=False
)

########################### launch ###########################
echo "=========================================="
echo "WebShop GRPO Training (MOCK Mode)"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Device: ${DEVICE}"
echo "GPUs: ${NGPUS_PER_NODE}"
echo "Batch size: ${train_batch_size}"
echo "Rollout N: ${rollout_n}"
echo "Learning rate: ${actor_lr}"
echo "NOTE: Using MOCK WebShop tools (no server required)"
echo "=========================================="

# Pre-flight checks
echo "Running pre-flight checks..."

if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    echo "Generating mock training data..."
    python3 examples/webshop_grpo/data_preprocess.py \
        --local_save_dir ${DATA_DIR} \
        --num_tasks 100 \
        --seed 42
fi

echo "Pre-flight checks passed!"
echo "Starting training..."

# Launch training
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "$@"
